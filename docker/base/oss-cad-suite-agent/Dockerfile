# poRTLe OSS CAD Suite Agent Image
#
# Combines the agent prerequisites from docker/base/agent-base (Python 3.12 from source,
# Node.js 22.x, common Python deps) with the full OSS CAD Suite toolchain setup from
# docker/base/oss-cad-suite.
#
# Use this as the parent for agents that need both the poRTLe agent stack and the
# complete open-source EDA toolchain (yosys/nextpnr/iverilog/verilator, etc.).

FROM ghcr.io/hdl/sim/osvb

WORKDIR /app

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    OSSCAD=/opt/oss-cad-suite

# Base system deps (Python build, OSSCAD install, Node.js setup)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        curl wget ca-certificates xz-utils gnupg \
        build-essential libssl-dev zlib1g-dev \
        libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
        gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf gdb-multiarch make; \
    rm -rf /var/lib/apt/lists/*

# Install OSS CAD Suite (same release as docker/base/oss-cad-suite)
RUN set -eux; \
    mkdir -p "$OSSCAD"; \
    curl -L \
      -o /tmp/oss-cad-suite.tgz \
      https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2025-10-31/oss-cad-suite-linux-x64-20251031.tgz; \
    tar -xzf /tmp/oss-cad-suite.tgz -C "$OSSCAD" --strip-components=1; \
    rm -f /tmp/oss-cad-suite.tgz
ENV PATH="$OSSCAD/bin:$PATH"

# Install Python 3.12 from source (agent-base behavior)
RUN set -eux; \
    wget https://www.python.org/ftp/python/3.12.5/Python-3.12.5.tgz; \
    tar -xvf Python-3.12.5.tgz; \
    cd Python-3.12.5; \
    ./configure --enable-optimizations; \
    make -j"$(nproc)"; \
    make altinstall; \
    cd ..; \
    rm -rf Python-3.12.5.tgz Python-3.12.5

# Upgrade pip and install common Python deps
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install --no-cache-dir requests pyyaml

# Install Node.js 22.x LTS (agent-base behavior)
RUN set -eux; \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -; \
    apt-get update; \
    apt-get install -y --no-install-recommends nodejs; \
    rm -rf /var/lib/apt/lists/*

# Default workdir for agents
WORKDIR /code
