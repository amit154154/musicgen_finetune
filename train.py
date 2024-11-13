from dataset.segments_dataset import MP3DataModule
from models.model import MusicGenLightningModule,AudioGenerationCallback
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning import seed_everything

# hyper params
segmented_audio_folder = "data"
save_recoding_dir = "123"
Duration = 8
batch_size = 12
learning_rate = 1e-4
num_epochs = 4        # Number of epochs to train
generate_every = 300
do_wandb = True
wandb_key = "wandb_key"
r = 8
accumulate_grad_batches = 1
seed = 42069  # nice , right ?
#seed_everything(seed)

if do_wandb:
    # Login to wandb (make sure you're logged in via command line or set the API key as an environment variable)
    wandb.login(key=wandb_key)
    # Initialize wandb run
    wandb_logger = WandbLogger(project="musicgen_maple_finetune_2",
    config={
        'duration': Duration,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'r':r
    })
else:
    wandb_logger = None


# ----- ModelCheckpoint Callback -----
# Initialize the ModelCheckpoint callback to save checkpoints every epoch
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints_real',                   # Directory to save checkpoints
    filename='musicgen-{epoch:02d}',         # Checkpoint filename format
    save_top_k=2,                            # Save top 2 checkpoints based on monitored metric
    monitor='val_loss',                      # Specify metric to monitor (e.g., validation loss)
    mode='min',                              # Use 'min' for metrics like loss, 'max' for accuracy
    every_n_epochs=1,                        # Save checkpoint every epoch
    save_last=True                           # Also save the last checkpoint
)

# ----- Training -----
data_module = MP3DataModule(
    data_dir=segmented_audio_folder,
    sample_rate=32000,
    duration=Duration,
    batch_size=batch_size,
    num_workers=0  # Set to the number of workers you desire
)
data_module.prepare_data()

musicgen_module = MusicGenLightningModule(learning_rate=learning_rate,log_wandb = do_wandb,r = r)

unique_conditions = data_module.unique_conditions
print(f"the prompts found are:{unique_conditions}")

# Add both AudioGenerationCallback and ModelCheckpoint to the callbacks list
callbacks = [
    AudioGenerationCallback(generate_every=generate_every, conditions=unique_conditions,save_recoding_dir = save_recoding_dir,do_wandb = do_wandb),
    checkpoint_callback  # Add the checkpoint callback
]


# ----- Trainer Configuration -----
trainer = pl.Trainer(
    max_epochs=num_epochs,
    logger=wandb_logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    callbacks=callbacks,
    log_every_n_steps=5,
    accumulate_grad_batches=accumulate_grad_batches,  # Add gradient accumulation
)

trainer.fit(musicgen_module, data_module)