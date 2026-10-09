#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
# 조각으로 나눈 학습 전사로 3-gram 언어모델을 만든다.
# 추가 단어 목록(의료 용어 등)은 한 줄에 하나씩 문장처럼 넣어 사전에 오르게 한다.
#
#   local/train_lm.sh data/train/text data/local/extra_units.txt data/local/lm

set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Usage: $0 <segmented-text> <extra-units|-> <lm-dir>"
  exit 1
fi
text=$1
extra=$2
dir=$3
order=${LM_ORDER:-3}

mkdir -p $dir
# 같은 문장을 여러 사람이 읽은 데이터라, 그대로 넣으면 언어모델이 그 문장들만
# 나온다고 배운다. 문장마다 한 번씩만 넣는다.
cut -d' ' -f2- $text | sort -u > $dir/corpus.txt
if [ "$extra" != "-" ] && [ -s "$extra" ]; then
  cat $extra >> $dir/corpus.txt
fi

python3 utils/lang/make_kn_lm.py -ngram-order $order \
  -text $dir/corpus.txt -lm $dir/lm.arpa
gzip -f $dir/lm.arpa
echo "$0: $(wc -l < $dir/corpus.txt) 줄로 ${order}-gram -> $dir/lm.arpa.gz"
