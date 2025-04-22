#!/bin/bash

# This script runs app by the user.
# Usage: sh app.sh [0|1|...] [use_float16]

# Get gpu_id
gpu_id=$1
if [ "$gpu_id" = "" ]; then
    gpu_id="0"
fi

# Use float 16
use_float16=$2

# Base command arguments
cmd_args=""

# Add use float 16
if [ "$use_float16" != "" ]; then
    cmd_args="$cmd_args --use_float16"
fi


# Run app
CUDA_VISIBLE_DEVICES=$gpu_id python3 app.py $cmd_args