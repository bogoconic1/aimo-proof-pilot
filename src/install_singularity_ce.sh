#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

SINGULARITY_VERSION="${SINGULARITY_VERSION:-v4.4.1}"
GO_VERSION="${GO_VERSION:-1.26.3}"
BUILD_ROOT="${SINGULARITY_BUILD_ROOT:-/tmp/singularity-build}"

if command -v singularity >/dev/null 2>&1; then
    singularity --version
    exit 0
fi

apt-get update
apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    ca-certificates \
    cryptsetup \
    fuse2fs \
    fuse3 \
    git \
    libfuse3-dev \
    libseccomp-dev \
    libtool \
    make \
    pkg-config \
    runc \
    squashfs-tools \
    squashfs-tools-ng \
    uidmap \
    wget \
    zlib1g-dev
rm -rf /var/lib/apt/lists/*

mkdir -p "${BUILD_ROOT}"
cd "${BUILD_ROOT}"

if ! command -v go >/dev/null 2>&1; then
    wget -q "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
    rm -rf /usr/local/go
    tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
fi

export PATH="/usr/local/go/bin:${PATH}"

if [ ! -d singularity/.git ]; then
    git clone --recurse-submodules https://github.com/sylabs/singularity.git
fi

git -C singularity fetch --depth 1 origin "${SINGULARITY_VERSION}" || true
git -C singularity checkout --recurse-submodules "${SINGULARITY_VERSION}"
git -C singularity submodule update --init --recursive

cd singularity
./mconfig --without-libsubid
make -C builddir -j"$(nproc)"
make -C builddir install

singularity --version
singularity buildcfg
