#!/bin/sh

# Build opencode-agent using central poRTLe opencode base image.
# Ensure central images are built first: cd ../../../docker && ./build.sh
#
# Override BASE_IMAGE to use a different base:
#   BASE_IMAGE=portle-agent-base:latest ./build_agent.sh

BASE_IMAGE=${BASE_IMAGE:-portle-opencode-base:latest}


echo "Using base image: $BASE_IMAGE"

docker build \
    --platform linux/amd64 \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f Dockerfile-agent \
    -t cvdp-opencode-agent \
    --no-cache \
    .

# For x86 emulation on M1/Apple Silicon, uncomment:
# docker build \
#     --platform linux/amd64 \
#     --build-arg BASE_IMAGE="$BASE_IMAGE" \
#     -f Dockerfile-agent \
#     -t cvdp-opencode-agent \
#     --no-cache \
#     .
