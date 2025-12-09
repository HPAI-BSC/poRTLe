# # poRTLe OSS CAD Suite Image
# #
# # Full open-source EDA toolchain image. Provides:
# # - iverilog (simulation)
# # - verilator (simulation)  
# # - yosys (synthesis)
# # - nextpnr (place & route)
# # - Python 3.12
# #
# # Use this for tasks that need the complete toolchain (simulation + synthesis + PnR).
# # For simulation-only tasks, use ghcr.io/hdl/sim/osvb instead.
# # For synthesis/PnR-only tasks, use ghcr.io/hdl/impl/pnr instead.

# FROM ghcr.io/hdl/sim/osvb

# WORKDIR /app

# # Install build dependencies
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         wget build-essential libssl-dev zlib1g-dev \
#         libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
#         libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
#         xz-utils \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python 3.12 from source
# RUN wget https://www.python.org/ftp/python/3.12.5/Python-3.12.5.tgz && \
#     tar -xvf Python-3.12.5.tgz && \
#     cd Python-3.12.5 && \
#     ./configure --enable-optimizations && \
#     make -j$(nproc) && \
#     make altinstall && \
#     cd .. && \
#     rm -rf Python-3.12.5.tgz Python-3.12.5

# # Upgrade pip
# RUN python3.12 -m pip install --upgrade pip

# # Install OSS CAD Suite (full toolchain: yosys, nextpnr, iverilog, verilator, etc.)
# # Using latest release (2025-11-30) for best SystemVerilog compatibility
# ENV OSSCAD=/opt/oss-cad-suite
# RUN mkdir -p "$OSSCAD" && \
#     curl -L -o /tmp/oss-cad-suite.tgz \
#         https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2025-11-30/oss-cad-suite-linux-x64-20251130.tgz && \
#     tar -xzf /tmp/oss-cad-suite.tgz -C "$OSSCAD" --strip-components=1 && \
#     rm /tmp/oss-cad-suite.tgz

# # Add OSS CAD Suite to PATH
# ENV PATH="$OSSCAD/bin:$PATH"

# # Install common Python dependencies
# RUN python3.12 -m pip install --no-cache-dir requests pyyaml cocotb pytest

FROM ghcr.io/hdl/sim/osvb
# FROM ghcr.io/yosyshq/oss-cad-suite
# FROM ghcr.io/hdl/containers:oss-cad-suite


WORKDIR /app

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl xz-utils ca-certificates \
    gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf gdb-multiarch make \
 && rm -rf /var/lib/apt/lists/*

# Install OSS CAD Suite
ENV OSSCAD=/opt/oss-cad-suite
RUN set -eux; \
    mkdir -p "$OSSCAD"; \
    curl -L \
      -o /tmp/oss-cad-suite.tgz \
      https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2025-10-31/oss-cad-suite-linux-x64-20251031.tgz; \
    tar -xzf /tmp/oss-cad-suite.tgz -C "$OSSCAD" --strip-components=1; \
    rm -f /tmp/oss-cad-suite.tgz
ENV PATH="$OSSCAD/bin:$PATH"


# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget build-essential libssl-dev zlib1g-dev \
	    libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
		libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
    && apt-get clean \
    && rm -rf /var/list/apt/lists/*
RUN wget https://www.python.org/ftp/python/3.12.5/Python-3.12.5.tgz
RUN tar -xvf Python-3.12.5.tgz
RUN cd Python-3.12.5 && \
    ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && rm -rf Python-3.12.5.tgz Python-3.12.5

# Upgrade pip
RUN python3.12 -m pip install --upgrade pip
RUN pip3.12 install --no-cache-dir requests pyyaml

