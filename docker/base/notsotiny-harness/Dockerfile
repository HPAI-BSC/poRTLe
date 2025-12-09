# poRTLe NotSoTiny Harness Image
#
# Based on the full OSS CAD Suite image, with benchmark-specific Python deps
# installed into the oss-cad-suite Python environment used by cocotb.

FROM portle-oss-cad-suite:latest

# Install benchmark-specific deps into the oss-cad Python env that cocotb uses
RUN /opt/oss-cad-suite/bin/tabbypy3 -m pip install --force-reinstall \
      "cocotb==1.9.1" "pytest==8.2.2"
