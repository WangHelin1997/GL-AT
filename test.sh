#!/bin/bash
DATASET_DIR="/data/dean/audioset"
WORKSPACE="/data/dean/audioset_tagging_cnn/workspaces"
GPU_ID=1
MODEL_TYPE='Cnn10'
BATCH_SIZE=32
N=5
LENGTH=2

CUDA_VISIBLE_DEVICES=$GPU_ID python3 pytorch/test.py audio_tagging --model_type=$MODEL_TYPE --audio_path="examples/R9_ZSCveAHg_7s.wav" --cuda