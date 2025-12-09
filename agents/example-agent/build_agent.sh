#!/bin/sh 

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build agent using central poRTLe base image.
# Ensure central images are built first: cd ../../docker && ./build.sh
#
# Override BASE_IMAGE to use a different base:
#   BASE_IMAGE=portle-oss-cad-suite:latest ./build_agent.sh

BASE_IMAGE=${BASE_IMAGE:-portle-agent-base:latest}

echo "Using base image: $BASE_IMAGE"

docker build \
    --platform linux/amd64 \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f Dockerfile-agent \
    -t cvdp-example-agent \
    --no-cache \
    .
