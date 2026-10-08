#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
# WER(어절 단위)과 CER(글자 단위)을 함께 낸다. 한국어는 띄어쓰기가 흔들려서
# WER만 보면 실제보다 나빠 보인다. 비교는 CER로 한다.

set -e -o pipefail
set -x
steps/scoring/score_kaldi_wer.sh "$@"
steps/scoring/score_kaldi_cer.sh --stage 2 "$@"
echo "$0: Done"
