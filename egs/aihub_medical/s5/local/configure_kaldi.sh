#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# Kaldi src/ 를 이 레시피에 맞는 설정으로 configure 한다.
# 긴 configure 명령을 터미널에 붙여넣다가 줄이 깨지는 일을 막으려고 스크립트로 둔다.
#
#   ~/bodeultext/egs/aihub_medical/s5/local/configure_kaldi.sh          # NVIDIA GPU (RTX 50 시리즈)
#   CUDA_ARCH="-gencode arch=compute_89,code=sm_89" local/configure_kaldi.sh  # 다른 GPU
#   USE_CUDA=no local/configure_kaldi.sh                                # GPU 없이

set -euo pipefail

root=$(cd "$(dirname "$0")/../../../.." && pwd)
openblas=$root/tools/OpenBLAS/install
cuda_dir=${CUDA_DIR:-/usr/local/cuda}
cuda_arch=${CUDA_ARCH:-"-gencode arch=compute_120,code=sm_120"}

if [ ! -f "$openblas/lib/libopenblas.so" ] && [ ! -f "$openblas/lib/libopenblas.a" ]; then
  echo "$0: OpenBLAS 가 없습니다: $openblas" >&2
  echo "     먼저: cd $root/tools && OPENBLAS_TARGET=HASWELL extras/install_openblas.sh" >&2
  exit 1
fi

cd "$root/src"
if [ "${USE_CUDA:-yes}" = no ]; then
  ./configure --shared --use-cuda=no --mathlib=OPENBLAS --openblas-root="$openblas"
else
  if [ ! -x "$cuda_dir/bin/nvcc" ]; then
    echo "$0: CUDA 를 못 찾았습니다: $cuda_dir/bin/nvcc (README 의 CUDA 설치 참고)" >&2
    exit 1
  fi
  ./configure --shared --use-cuda --cudatk-dir="$cuda_dir" --cuda-arch="$cuda_arch" \
    --mathlib=OPENBLAS --openblas-root="$openblas"
fi
