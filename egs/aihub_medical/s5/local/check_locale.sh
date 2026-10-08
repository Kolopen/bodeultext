#!/usr/bin/env bash
# Kaldi 의 utils/validate_data_dir.sh 는 en_US.UTF-8(또는 C.UTF-8) 로케일로
# 전사에 깨진 글자가 있는지 본다. 그 로케일이 없으면 한글이 전부 깨진 글자로
# 보여서 중간에 멈춘다. 새로 깐 WSL Ubuntu 가 이 상태다. 시작 전에 확인한다.
L=en_US.UTF-8
locale -a 2>/dev/null | grep -q "C.UTF-8" && L=C.UTF-8
n=$(echo "가나다" | LC_ALL=$L grep -c '[^[:print:][:space:]]' || true)
if [ "$n" != 0 ]; then
  echo "$0: 이 컴퓨터에 $L 로케일이 없어 Kaldi 가 한글을 깨진 글자로 봅니다." >&2
  echo "     Ubuntu(WSL) 에서 한 번만 실행하세요:  sudo locale-gen en_US.UTF-8" >&2
  exit 1
fi
