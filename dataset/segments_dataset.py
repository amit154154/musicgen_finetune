import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pytorch_lightning as pl
from tqdm import tqdm  # For the progres

class RandomSegmentMP3Dataset(Dataset):
    def __init__(self, files, conditions, sample_rate=32000, duration=12, is_validation=False,
                 samples_train = 50000,samples_val = 10000,train_ratio = 0.8):
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_validation = is_validation
        self.samples_train = samples_train
        self.samples_val = samples_val
        self.train_ratio = train_ratio

        self.files = files
        self.conditions = conditions

        # Calculate the number of samples per segment
        self.num_samples = int(self.sample_rate * self.duration)

    def __len__(self):
        # Adjusted length for training/validation
        if self.is_validation:
            return self.samples_val  # Adjust as needed
        else:
            return self.samples_train   # Adjust as needed

    def __getitem__(self, idx):
        # Randomly select a file
        file_idx = random.randint(0, len(self.files) - 1)
        filepath = self.files[file_idx]
        condition = self.conditions[file_idx]

        # Get audio info dynamically to avoid preloading
        info = torchaudio.info(filepath)
        total_frames = info.num_frames

        if self.is_validation:
            start_limit = max(int(total_frames * self.train_ratio), 0)
            end_limit = total_frames - self.num_samples
            if end_limit <= start_limit:
                start_limit = max(end_limit - 1, 0)
        else:
            start_limit = 0
            end_limit = max(int(total_frames * self.train_ratio) - self.num_samples, 0)
            if end_limit <= start_limit:
                end_limit = start_limit + 1

        if end_limit <= start_limit:
            num_frames = min(total_frames, self.num_samples)
            waveform, sr = torchaudio.load(filepath, frame_offset=0, num_frames=num_frames)
        else:
            start_frame = random.randint(start_limit, end_limit)
            waveform, sr = torchaudio.load(filepath, frame_offset=start_frame, num_frames=self.num_samples)

        # Pad waveform if it's shorter than self.num_samples
        if waveform.size(1) < self.num_samples:
            padding = self.num_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Ensure waveform has exact length self.num_samples
        waveform = waveform[:, :self.num_samples]

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform, condition  # Return both segment and condition

# ----- DataModule Class -----
class MP3DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, sample_rate=32000, duration=12, batch_size=4, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.files = []
        self.conditions = []

    def prepare_data(self):
        # Collect all MP3 file paths and their conditions
        print("Collecting MP3 file paths and conditions...")
        for root, dirs, files in os.walk(self.data_dir):
            for file in tqdm(files):
                if file.endswith('.mp3'):
                    filepath = os.path.join(root, file)
                    # The condition is the immediate subfolder of data_dir
                    rel_path = os.path.relpath(root, self.data_dir)
                    condition = os.path.normpath(rel_path).split(os.sep)[0]
                    self.files.append(filepath)
                    self.conditions.append(condition)

        # Collect all unique conditions for generation
        self.unique_conditions = list(set(self.conditions))

    def setup(self, stage=None):
        # Split data into training and validation
        self.train_dataset = RandomSegmentMP3Dataset(
            files=self.files,
            conditions=self.conditions,
            sample_rate=self.sample_rate,
            duration=self.duration,
            is_validation=False
        )

        self.val_dataset = RandomSegmentMP3Dataset(
            files=self.files,
            conditions=self.conditions,
            sample_rate=self.sample_rate,
            duration=self.duration,
            is_validation=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False  # Avoid dropping last batch for consistent batch size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False  # Avoid dropping last batch for consistent batch size
        )