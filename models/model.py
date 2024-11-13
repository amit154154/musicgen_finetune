import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from tqdm import tqdm  # For the progress bar
import torch.nn as nn
from peft import LoraConfig, get_peft_model  # Import PEFT components
from pytorch_optimizer import SOAP

from torch.optim.lr_scheduler import LambdaLR

class MusicGenLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, r=4, log_wandb=False, warmup_pct=0.1):
        super().__init__()
        # Load pre-trained MusicGen model
        self.musicgen_model = MusicGen.get_pretrained('facebook/musicgen-small')
        self.lm = self.musicgen_model.lm
        self.learning_rate = learning_rate
        self.log_wandb = log_wandb
        self.warmup_pct = warmup_pct

        # Define LoRA configuration for the transformer
        def find_linear_module_names(model):
            target_types = (nn.Linear, nn.Conv1d, nn.Conv2d)  # Adjust as needed
            linear_module_names = []
            for name, module in model.named_modules():
                if isinstance(module, target_types):
                    linear_module_names.append(name)
            return linear_module_names

        linear_module_names = find_linear_module_names(self.lm)
        lora_config = LoraConfig(
            r=r,
            lora_alpha=32,
            target_modules=linear_module_names,
            lora_dropout=0.05,
            bias="none",
        )
        self.lm = get_peft_model(self.lm, lora_config)
        self.lm.float()

        # Use CrossEntropyLoss for computing loss
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, codes, attributes):
        lm_output = self.lm.compute_predictions(codes=codes, conditions=attributes)
        return lm_output

    def training_step(self, batch, batch_idx):
        segments, conditions = batch
        batch = segments

        with torch.no_grad():
            tokens, _ = self.musicgen_model.compression_model.encode(batch)

        attributes = [ConditioningAttributes(text={'description': condition}) for condition in conditions]
        lm_output = self.lm.compute_predictions(codes=tokens, conditions=attributes)
        logits = lm_output.logits
        mask = lm_output.mask

        B, K, T, card = logits.shape
        logits = logits.permute(0, 2, 1, 3).reshape(-1, card)
        target_tokens = tokens.permute(0, 2, 1).reshape(-1)
        mask = mask.permute(0, 2, 1).reshape(-1)

        valid_indices = mask.bool()
        logits = logits[valid_indices]
        target_tokens = target_tokens[valid_indices]

        loss = self.criterion(logits, target_tokens)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log the learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        segments, conditions = batch
        batch = segments

        with torch.no_grad():
            tokens, _ = self.musicgen_model.compression_model.encode(batch)

        attributes = [ConditioningAttributes(text={'description': condition}) for condition in conditions]
        lm_output = self.lm.compute_predictions(codes=tokens, conditions=attributes)
        logits = lm_output.logits
        mask = lm_output.mask

        B, K, T, card = logits.shape
        logits = logits.permute(0, 2, 1, 3).reshape(-1, card)
        target_tokens = tokens.permute(0, 2, 1).reshape(-1)
        mask = mask.permute(0, 2, 1).reshape(-1)

        valid_indices = mask.bool()
        logits = logits[valid_indices]
        target_tokens = target_tokens[valid_indices]

        val_loss = self.criterion(logits, target_tokens)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.lm.parameters(), lr=self.learning_rate)

        # Scheduler with linear warmup for 10% of total steps
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(self.warmup_pct * num_training_steps)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]


# ----- Audio Generation Callback -----
class AudioGenerationCallback(pl.Callback):
    def __init__(self, generate_every=100, conditions=None, save_recoding_dir='', do_wandb=False):
        self.generate_every = generate_every
        self.global_step = 0
        self.conditions = conditions  # List of conditions
        self.save_recoding_dir = save_recoding_dir
        self.do_wandb = do_wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.global_step += 1
        if self.global_step % self.generate_every == 0:
            model = pl_module.musicgen_model
            model.lm.eval()  # Set model to evaluation mode for generation
            with torch.no_grad():
                # Generate audio for each unique condition
                for condition in self.conditions:
                    # Generate audio
                    generated_audio = model.generate(descriptions=[condition], progress=False)
                    # Save generated audio
                    generated_audio_filename = f'{self.save_recoding_dir}/generated_{condition}_step{self.global_step}.wav'
                    torchaudio.save(generated_audio_filename, generated_audio[0].cpu(), sample_rate=32000)

                    # Log generated audio to wandb if enabled
                    if self.do_wandb:
                        trainer.logger.experiment.log({
                            f'generated_audio_{condition}': wandb.Audio(generated_audio_filename, sample_rate=32000),
                            'step': self.global_step
                        })

                    print(
                        f"Saved generated audio for condition '{condition}' at step {self.global_step}: {generated_audio_filename}")
            model.lm.train()  # Set model back to training mode