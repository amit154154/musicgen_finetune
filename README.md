# MusicGen Genre Fine-Tuning with LoRA

This repository provides tools and scripts to fine-tune Meta's [MusicGen](https://github.com/facebookresearch/audiocraft) model using Low-Rank Adaptation (LoRA). The goal is to adapt MusicGen to generate music in specific genres by conditioning it on genre-related textual descriptions.

## Overview

[MusicGen](https://github.com/facebookresearch/audiocraft) is a state-of-the-art, controllable text-to-music model developed by Meta. It utilizes a single-stage auto-regressive Transformer architecture trained over a 32kHz EnCodec tokenizer with four codebooks sampled at 50 Hz. By fine-tuning MusicGen with LoRA, we can efficiently adapt the model to generate music in desired genres without retraining the entire model,
and still conserve the original model's controllable generation capabilities.


