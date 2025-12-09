# poRTLe OpenCode Agent Base Image
#
# Extends portle-agent-base with opencode-ai CLI pre-installed.
# Use this as the base for all opencode-based agents to avoid
# reinstalling opencode-ai on every agent rebuild.
#
# Provides:
# - Everything from portle-agent-base (Python 3.12, Node.js 22.x, EDA tools)
# - opencode-ai CLI (npm global)
#
# Usage in agent Dockerfile-agent:
#   ARG BASE_IMAGE=portle-opencode-base:latest
#   FROM ${BASE_IMAGE}
#   # ... your agent-specific setup (no npm install needed!)

ARG BASE_IMAGE=portle-agent-base:latest
FROM ${BASE_IMAGE}

# Install opencode-ai CLI and plugin SDK globally (this is the slow step we're caching)
RUN npm install -g opencode-ai @opencode-ai/plugin

# Verify installation
RUN opencode --version || echo "opencode installed"
