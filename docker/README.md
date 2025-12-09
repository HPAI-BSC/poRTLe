# poRTLe Docker Images

Central Docker image definitions for poRTLe agents and benchmark runners.

## Quick Start

```bash
# Build all central images
cd docker
./build.sh

# Build a specific image
./build.sh agent-base
./build.sh oss-cad-suite
```

## Available Images

### Central Images (built locally)

| Image | Tag | Purpose |
|-------|-----|---------|
| Agent Base | `portle-agent-base:latest` | Base for all agents (Python 3.12 + Node.js) |
| OpenCode Base | `portle-opencode-base:latest` | Base for opencode agents (includes opencode-ai CLI) |
| OSS CAD Suite | `portle-oss-cad-suite:latest` | Full EDA toolchain (iverilog + yosys + nextpnr) |

### External Images (pulled from registries)

| Image | Tag | Purpose |
|-------|-----|---------|
| Simulation | `ghcr.io/hdl/sim/osvb` | iverilog, verilator, cocotb |
| Synthesis/PnR | `ghcr.io/hdl/impl/pnr` | yosys, nextpnr (no iverilog!) |

## Usage

### For Agents

Agents should inherit from `portle-agent-base` instead of duplicating the Python/Node.js setup:

```dockerfile
# Dockerfile-base (minimal - just use ARG)
ARG BASE_IMAGE=portle-agent-base:latest
FROM ${BASE_IMAGE}

# Add any agent-specific base dependencies here
```

```dockerfile
# Dockerfile-agent
FROM my-agent-base

# Install LLM tooling here (opencode, codex, etc.)
RUN npm install -g opencode-ai

# Copy agent code
COPY agent.py .
```

### For Tasks (docker-compose.yml)

Use template placeholders that get substituted by the benchmark runner:

```yaml
services:
  test:
    image: __OSS_SIM_IMAGE__  # Substituted to ghcr.io/hdl/sim/osvb
    # ...

  synth:
    image: __PORTLE_OSS_CAD_IMAGE__  # Substituted to portle-oss-cad-suite:latest
    # ...
```

### Template Placeholders

| Placeholder | Default Value | Use Case |
|-------------|---------------|----------|
| `__OSS_SIM_IMAGE__` | `ghcr.io/hdl/sim/osvb` | Simulation tasks |
| `__OSS_PNR_IMAGE__` | `ghcr.io/hdl/impl/pnr` | Synthesis/PnR tasks |
| `__PORTLE_AGENT_BASE__` | `portle-agent-base:latest` | Agent base image |
| `__PORTLE_OSS_CAD_IMAGE__` | `portle-oss-cad-suite:latest` | Full toolchain tasks |

Override defaults in `.env`:

```bash
PORTLE_OSS_CAD_IMAGE=my-custom-eda:v2
```

## Build Order

Central images must be built before agent images that depend on them:

1. `./docker/build.sh` (builds central images)
2. `./agents/example-agent/build_agent.sh` (builds agent using central base)

## Directory Structure

```
docker/
├── README.md           # This file
├── build.sh            # Build script for all central images
├── images.yaml         # Image registry manifest
└── base/
    ├── agent-base/
    │   └── Dockerfile  # Python 3.12 + Node.js base
    └── oss-cad-suite/
        └── Dockerfile  # Full EDA toolchain
```
