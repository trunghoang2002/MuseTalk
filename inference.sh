#!/bin/bash

# This script runs inference based on the version and mode specified by the user.
# Usage:
# To run v1.0 inference: sh inference.sh v1.0 [normal|realtime] [0|1|...] [use_float16]
# To run v1.5 inference: sh inference.sh v1.5 [normal|realtime] [0|1|...] [use_float16]

# Check if the correct number of arguments is provided
if [ "$#" -gt 4 ]; then
    echo "Too many arguments."
    echo "Usage: $0 <version> <mode> <gpu_id> <use_float16 (optional)>"
    echo "Example: $0 v1.0 normal 0 use_float16 or $0 v1.5 realtime 1"
    exit 1
fi

# Get the version and mode from the user input
version=$1
mode=$2

# Get gpu_id
gpu_id=$3

# Use float 16
use_float16=$4

# Validate mode
if [ "$mode" != "normal" ] && [ "$mode" != "realtime" ]; then
    echo "Invalid mode specified. Please use 'normal' or 'realtime'."
    exit 1
fi

# Set config path based on mode
if [ "$mode" = "normal" ]; then
    config_path="./configs/inference/test.yaml"
    result_dir="./results/test"
else
    config_path="./configs/inference/realtime.yaml"
    result_dir="./results/realtime"
fi

# Define the model paths based on the version
if [ "$version" = "v1.0" ]; then
    model_dir="./models/musetalk"
    unet_model_path="$model_dir/pytorch_model.bin"
    unet_config="$model_dir/musetalk.json"
    version_arg="v1"
elif [ "$version" = "v1.5" ]; then
    model_dir="./models/musetalkV15"
    unet_model_path="$model_dir/unet.pth"
    unet_config="$model_dir/musetalk.json"
    version_arg="v15"
else
    echo "Invalid version specified. Please use v1.0 or v1.5."
    exit 1
fi

# Set script name based on mode
if [ "$mode" = "normal" ]; then
    script_name="scripts.inference"
else
    script_name="scripts.realtime_inference"
fi

# Base command arguments
cmd_args="--inference_config $config_path \
    --result_dir $result_dir \
    --unet_model_path $unet_model_path \
    --unet_config $unet_config \
    --version $version_arg"

# Add realtime-specific arguments if in realtime mode
if [ "$mode" = "realtime" ]; then
    cmd_args="$cmd_args \
    --fps 25 \
    --version $version_arg"
fi

# Add use float 16
if [ "$use_float16" != "" ] && [ "$mode" = "normal" ]; then
    cmd_args="$cmd_args \
    --use_float16"
fi

# Run inference
CUDA_VISIBLE_DEVICES=$gpu_id python3 -m $script_name $cmd_args
