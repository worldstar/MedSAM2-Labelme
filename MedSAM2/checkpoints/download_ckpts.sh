#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Define the URLs for the checkpoints
BASE_URL="https://huggingface.co/jiayuanz3/MedSAM2_pretrain/resolve/main/"
medsam2_url="${BASE_URL}MedSAM2_pretrain.pth"


# Download each of the four checkpoints using wget
echo "Downloading medsam2_pretrain.pth checkpoint..."
wget $medsam2_url || { echo "Failed to download checkpoint from $medsam2_url"; exit 1; }

echo "All checkpoints are downloaded successfully."
