# MusicGen Genre Fine-Tuning with LoRA

This repository provides tools and scripts to fine-tune Meta's [MusicGen](https://github.com/facebookresearch/audiocraft) model using Low-Rank Adaptation (LoRA). The goal is to adapt MusicGen to generate music in specific genres by conditioning it on genre-related textual descriptions.

## Overview

[MusicGen](https://github.com/facebookresearch/audiocraft) is a state-of-the-art, controllable text-to-music model developed by Meta. It utilizes a single-stage auto-regressive Transformer architecture trained over a 32kHz EnCodec tokenizer with four codebooks sampled at 50 Hz. By fine-tuning MusicGen with LoRA, we can efficiently adapt the model to generate music in desired genres without retraining the entire model, and still conserve the original model's controllable generation capabilities.

## Evaluation

To evaluate the effectiveness of the fine-tuning process, we compared the generated audio against the ground truth MapleStory background music using the **Frechet Distance** metric. The Frechet Distance measures the similarity between two distributions, where a lower score indicates higher similarity.

### Frechet Distance Results

We conducted evaluations on **150 generated audio files**, each **8 seconds** in duration. Below are the results comparing the zero-shot performance with the fine-tuned model:

| **prompt**                                                                                                                                                                    | **Frechet Distance** |      
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| **Zero-Shot Generation:**                                                                                                                                                     |                      |
| *"maplestory background music"*                                                                                                                          | 1.0153               |
| *"upbeat orchestral piece with jazz influences, featuring piano and strings, capturing the whimsical and adventurous spirit of a fantasy world"*         | 0.7420               |
| *"electronic track with a lighthearted and playful mood, incorporating synthesizers and woodwind instruments, suitable for a vibrant game environment."* | 0.6013               |
