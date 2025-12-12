# Agents

> **Note:** The contents of this directory are **not tracked by git** (see `.gitignore`). Clone external agent projects into this directory to use them with poRTLe.

You can drop in **any agent implementation** you already have—LLM-based or not. The only contract for CVDP benchmarks is that the container runs `agent.py` and that your code can edit files in `/rtl` (and other task folders). Your agent can be:
- A hand-written deterministic script.
- Built on frameworks (e.g., LangGraph) or general-purpose coding agents (e.g., OpenCode, Claude, etc.).
- Fully custom scratch code.

Bundle everything the agent needs (code, configs, model clients, libraries) into the agent folder and its `Dockerfile-agent` so the container has all dependencies at runtime.

## Overview

This directory contains AI agent implementations that can be tested against benchmarks. Each agent should be a self-contained project directory.

## Available Agents

Clone the following agent projects into this directory for use with poRTLe:

### Example Agent Template

```text
**Agent Name:** [Your Agent Name]
**Repository:** [GitHub/GitLab URL]
**Commit:** [commit-hash] (for compatibility)
**Description:** [Brief description]
**Setup:**
  cd agents/
  git clone [repo-url] [agent-name]
  cd [agent-name]
  git checkout [commit-hash]
  [additional setup commands]
```

---

## Docker Base Images

poRTLe provides centralized Docker base images that agents should inherit from.
These are located in `docker/base/` and built automatically when you run benchmarks.

### Available Base Images

| Image | Tag | Purpose |
|-------|-----|---------|
| Agent Base | `portle-agent-base:latest` | Python 3.12 + Node.js 22.x for LLM agents |
| OSS CAD Suite | `portle-oss-cad-suite:latest` | Full EDA toolchain (simulation + synthesis + PnR) |

### Building Base Images Manually

```bash
# Build all central images
./docker/build.sh

# Build specific image
./docker/build.sh agent-base
./docker/build.sh oss-cad-suite
```

### Using Base Images in Your Agent

Agents should use a **single Dockerfile** that inherits from the central base image.
Here's the recommended pattern:

```dockerfile
# Dockerfile-agent
ARG BASE_IMAGE=portle-agent-base:latest
FROM ${BASE_IMAGE}

# Install agent-specific tools
RUN npm install -g your-agent-cli

# Copy agent code
WORKDIR /agent
COPY . .

# Install Python dependencies if needed
RUN pip install -r requirements.txt

# Set entry point
CMD ["your-agent-command"]
```

The `BASE_IMAGE` argument allows flexibility:
- Default: Uses the central `portle-agent-base:latest`
- Override: Pass `--build-arg BASE_IMAGE=custom:tag` during build

### Template Placeholders for docker-compose.yml

When your agent needs simulation/synthesis tools, use these placeholders in
`docker-compose.yml`:

| Placeholder | Image | Use Case |
|-------------|-------|----------|
| `__OSS_SIM_IMAGE__` | `hdlc/sim:osvb` | Simulation only (iverilog, verilator, cocotb) |
| `__OSS_PNR_IMAGE__` | `hdlc/impl:pnr` | Synthesis + PnR (yosys, nextpnr) |
| `__PORTLE_AGENT_BASE__` | `portle-agent-base:latest` | Agent base image |
| `__PORTLE_OSS_CAD_IMAGE__` | `portle-oss-cad-suite:latest` | Full EDA (simulation + synthesis + PnR) |

Example `docker-compose.yml`:

```yaml
services:
  harness:
    image: __PORTLE_OSS_CAD_IMAGE__
    volumes:
      - ./shared:/workspace/shared
    working_dir: /workspace
```

poRTLe automatically substitutes these placeholders at runtime.

---

## Adding Your Own Agent

1. Clone your agent repository into this directory
2. Ensure it has the required interface for your benchmark
3. Add it to the database: `python src/add_agent.py <agent_id>`
4. Document it in this README with repository URL and compatible commit hash

---

## Agent Directory Structure

Recommended structure for agent directories:

```text
agents/
├── your-agent/
│   ├── Dockerfile-agent      # Single Dockerfile inheriting from base
│   ├── build_agent.sh        # Build script (optional)
│   ├── portle_config.json    # Agent configuration for poRTLe
│   ├── requirements.txt      # Python dependencies
│   └── ...                   # Agent source code
```

### portle_config.json Example

```json
{
  "agent_name": "your-agent",
  "docker_image": "your-agent:latest",
  "build_context": ".",
  "dockerfile": "Dockerfile-agent",
  "timeout_seconds": 3600
}
```

---

## Built-in Agents

### example-agent

A minimal example demonstrating the agent interface. Use this as a template
for creating new agents.

### opencode-agent

An agent using the [OpenCode](https://github.com/opencode-ai/opencode) CLI
for code generation tasks. Includes configuration for various LLM providers.
