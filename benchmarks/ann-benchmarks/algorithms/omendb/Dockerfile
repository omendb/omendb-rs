FROM ann-benchmarks

# Install Rust (needed to build omendb from source)
RUN apt-get update && apt-get install -y curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin for building Python wheels
RUN pip install maturin

# Install omendb from PyPI (when published)
# RUN pip install omendb

# For development: copy and install from local wheel
# COPY ./omendb-*.whl /tmp/
# RUN pip install /tmp/omendb-*.whl

# For development: build from source
COPY ./omendb /tmp/omendb
RUN cd /tmp/omendb/python && maturin build --release && pip install /tmp/omendb/target/wheels/*.whl

# Verify installation
RUN python -c 'import omendb; print(f"OmenDB version: {omendb.__version__}")'
