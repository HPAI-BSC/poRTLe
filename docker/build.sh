#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build central poRTLe Docker images
# 
# Usage:
#   ./build.sh              # Build all images
#   ./build.sh agent-base                # Build only agent-base
#   ./build.sh oss-cad-suite             # Build only oss-cad-suite
#   ./build.sh oss-cad-suite-agent       # Build only oss-cad-suite-agent
#   ./build.sh notsotiny-harness         # Build only notsotiny-harness

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

build_image() {
    local name="$1"
    local tag="$2"
    local dockerfile="$SCRIPT_DIR/base/$name/Dockerfile"
    
    if [[ ! -f "$dockerfile" ]]; then
        echo "Error: Dockerfile not found at $dockerfile"
        return 1
    fi
    
    echo "=========================================="
    echo "Building $tag"
    echo "=========================================="
    
    docker build --platform linux/amd64 -f "$dockerfile" -t "$tag" "$SCRIPT_DIR/base/$name"
    
    echo "✓ Built $tag"
    echo ""
}

# Image definitions (order matters - dependencies first)
IMAGES="agent-base:portle-agent-base:latest
opencode-base:portle-opencode-base:latest
oss-cad-suite:portle-oss-cad-suite:latest
notsotiny-harness:portle-notsotiny-harness:latest
oss-cad-suite-agent:portle-oss-cad-suite-agent:latest"

# Main
if [[ $# -eq 0 ]]; then
    # Build all images
    echo "Building all poRTLe base images..."
    echo ""
    echo "$IMAGES" | while IFS=: read -r name tag; do
        build_image "$name" "$tag"
    done
    echo "=========================================="
    echo "All images built successfully!"
    echo ""
    echo "Available images:"
    echo "$IMAGES" | while IFS=: read -r name tag; do
        echo "  - $tag"
    done
else
    # Build specific image
    found=false
    echo "$IMAGES" | while IFS=: read -r name tag; do
        if [[ "$name" == "$1" ]]; then
            build_image "$name" "$tag"
            found=true
        fi
    done
    
    if [[ "$found" != "true" ]]; then
        # Check again outside the subshell
        match=$(echo "$IMAGES" | grep "^$1:")
        if [[ -z "$match" ]]; then
            echo "Error: Unknown image '$1'"
            echo ""
            echo "Available images:"
            echo "$IMAGES" | while IFS=: read -r name tag; do
                echo "  - $name ($tag)"
            done
            exit 1
        else
            name=$(echo "$match" | cut -d: -f1)
            tag=$(echo "$match" | cut -d: -f2-)
            build_image "$name" "$tag"
        fi
    fi
fi
