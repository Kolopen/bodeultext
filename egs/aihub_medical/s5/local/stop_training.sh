#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# 돌고 있는 학습을 딸린 작업까지 통째로 멈춘다.
#
#   local/stop_training.sh          # 무엇을 멈출지 보여 주고 물어본다
#   local/stop_training.sh --yes    # 묻지 않고 멈춘다
#
# nohup ... & 로 띄운 학습은 Ctrl+C 로 멈추지 않는다. run.sh 만 끄면 그 아래에서
# 돌던 Kaldi 프로그램(nnet3-chain-train, gmm-est 등)이 남아서 계속 돈다.
# 그래서 run.sh, run_tdnn.sh, expand_lm.sh 와 그 자식들을 모두 찾아 끈다.
# 멈춘 뒤 다시 시작하는 법은 다음단계.txt 의 [3].

set -u

pattern='(^|[ /])(run|run_tdnn|run_ivector_common|expand_lm)\.sh( |$)'
# 이 스크립트를 부른 셸과 그 위는 건드리지 않는다.
mine=" $$ "
p=$$
while [ "$p" -gt 1 ] 2>/dev/null; do
  p=$(ps -o ppid= -p "$p" | tr -d ' ')
  mine="$mine$p "
done
roots=$(for p in $(pgrep -f "$pattern"); do
  case "$mine" in *" $p "*) ;; *) echo "$p" ;; esac
done)
if [ -z "$roots" ]; then
  echo "돌고 있는 학습이 없습니다."
  exit 0
fi

descendants() {
  local c
  for c in $(pgrep -P "$1"); do
    descendants "$c"
    echo "$c"
  done
}
all=$(for p in $roots; do descendants "$p"; echo "$p"; done | sort -un | grep -vx "$$")

echo "멈출 작업 ($(echo $all | wc -w)개):"
ps -o pid=,etime=,args= -p "$(echo $all | tr ' ' ,)" | cut -c1-110 | sed 's/^/  /'
if [ "${1:-}" != "--yes" ]; then
  read -r -p "멈출까요? [y/N] " answer
  case "$answer" in y|Y|yes) ;; *) echo "그대로 둡니다."; exit 0 ;; esac
fi

kill -TERM $all 2>/dev/null
for _ in 1 2 3 4 5 6 7 8 9 10; do
  alive=$(ps -o pid= -p "$(echo $all | tr ' ' ,)" 2>/dev/null)
  [ -z "$alive" ] && break
  sleep 1
done
if [ -n "${alive:-}" ]; then
  kill -KILL $alive 2>/dev/null
fi

echo "멈췄습니다."
echo "로그 끝을 보고 어느 단계였는지 확인한 뒤 다음단계.txt [3] 대로 이어서 하세요."
latest=$(ls -t exp/chain/tdnn1a/[0-9]*.mdl 2>/dev/null | head -1)
# final.mdl 이 더 새것이면 신경망 학습은 이미 끝난 것이다.
if [ -n "$latest" ] && ! [ exp/chain/tdnn1a/final.mdl -nt "$latest" ]; then
  n=$(basename "$latest" .mdl)
  echo "신경망은 $n 번째 반복까지 저장돼 있습니다. 이어서 하려면:"
  echo "  nohup local/chain/run_tdnn.sh --nj 4 --stage 12 --train-stage $n > run_chain.log 2>&1 &"
fi
