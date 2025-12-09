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

# Build base layer (installs opencode + requirements on top of central image)
docker build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f Dockerfile-base \
    -t cvdp-opencode-agent-base \
    .

# Build agent layer (copies agent code)
docker build \
    -f Dockerfile-agent \
    -t cvdp-opencode-agent \
    --no-cache \
    .

# For Apple Silicon (M1/M2), uncomment and use:
# docker build --platform linux/amd64 --build-arg BASE_IMAGE="$BASE_IMAGE" -f Dockerfile-base -t cvdp-opencode-agent-base .
# docker build --platform linux/amd64 -f Dockerfile-agent -t cvdp-opencode-agent --no-cache .
