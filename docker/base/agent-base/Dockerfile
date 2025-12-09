# poRTLe Agent Base Image
# 
# Base image for all poRTLe agents. Provides:
# - Python 3.12 (compiled from source)
# - Node.js 22.x LTS
# - Common Python dependencies
#
# Agents should inherit from this and add their specific LLM tooling
# (opencode, codex, etc.) in their own Dockerfile-agent layer.

FROM ghcr.io/hdl/sim/osvb

WORKDIR /app

# Install build dependencies for Python compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget build-essential libssl-dev zlib1g-dev \
        libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 from source
RUN wget https://www.python.org/ftp/python/3.12.5/Python-3.12.5.tgz && \
    tar -xvf Python-3.12.5.tgz && \
    cd Python-3.12.5 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.12.5.tgz Python-3.12.5

# Upgrade pip
RUN python3.12 -m pip install --upgrade pip

# Install Node.js 22.x LTS
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install common Python dependencies
RUN python3.12 -m pip install --no-cache-dir requests pyyaml
