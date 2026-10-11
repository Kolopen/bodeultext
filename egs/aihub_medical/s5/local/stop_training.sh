#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# 돌고 있는 학습을 딸린 작업까지 통째로 멈춘다.
#
#   local/stop_training.sh          # 무엇을 멈출지 보여 주고 물어본다
#   local/stop_training.sh --list   # 돌고 있는 학습만 보여 준다 (멈추지 않는다)
#   local/stop_training.sh --yes    # 묻지 않고 멈춘다
#
# nohup ... & 로 띄운 학습은 Ctrl+C 로 멈추지 않는다. run.sh 만 끄면 그 아래에서
# 돌던 Kaldi 프로그램(nnet3-chain-train, gmm-est 등)이 남아서 계속 돈다.
# 그래서 이 폴더에서 띄운 run.sh, run_tdnn.sh, expand_lm.sh 와 그 자식들, 그리고
# 같은 묶음(프로세스 그룹)에 있는 작업을 모두 찾아 끈다. 묻는 동안 새로 뜬 작업도
# 잡도록 끄기 직전에 다시 찾고, 남은 것이 없을 때까지 되풀이한다.
# 멈춘 뒤 다시 시작하는 법은 다음단계.txt 의 [3].

set -u

here=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$here" || exit 1

# 학습 대본은 '#!/usr/bin/env bash' 라 'bash ./run.sh ...' 꼴로 뜬다. 편집기(nano run.sh)나
# 다른 저장소의 run.sh(-AI 의 scripts/run.sh)는 고르지 않도록 실행 폴더도 본다.
pattern='^(/usr/bin/|/bin/)?(ba)?sh ([^ ]*/)?(run|run_tdnn|run_ivector_common|expand_lm)\.sh( |$)'
# 학습 대본이 먼저 죽어 홀로 남은 Kaldi 작업 (녹음 인식용 online2-, lattice- 는 뺀다)
workers='run\.pl|steps/|utils/|nnet3-|gmm-|ivector-|train\.py|compute-mfcc|ali-to-|align-'

# 이 스크립트를 부른 셸과 그 위는 건드리지 않는다.
mine=" $$ "
p=$$
while [ "$p" -gt 1 ] 2>/dev/null; do
  p=$(ps -o ppid= -p "$p" | tr -d ' ')
  mine="$mine$p "
done

in_here() { [ "$(readlink "/proc/$1/cwd" 2>/dev/null)" = "$here" ]; }
not_mine() { case "$mine" in *" $1 "*) return 1 ;; esac; return 0; }

descendants() {
  local c
  for c in $(pgrep -P "$1"); do
    descendants "$c"
    echo "$c"
  done
}

targets() {
  local p g roots=
  for p in $(pgrep -f "$pattern"); do
    not_mine "$p" && in_here "$p" && roots="$roots $p"
  done
  {
    for p in $roots; do
      descendants "$p"
      echo "$p"
      # nohup ... & 로 띄웠으면 그 학습이 묶음의 우두머리다. 묶음째 끈다.
      g=$(ps -o pgid= -p "$p" | tr -d ' ')
      [ "$g" = "$p" ] && pgrep -g "$g"
    done
    for p in $(pgrep -f "$workers"); do
      # 녹음 인식(report_audio.sh)도 이 폴더에서 nnet3 프로그램을 돌린다. 학습이 아니다.
      case "$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)" in *online2-wav-*) continue ;; esac
      in_here "$p" && echo "$p"
    done
  } | sort -un | while read -r p; do
    not_mine "$p" && [ -d "/proc/$p" ] && echo "$p"
  done
}

show() {
  ps -o pid=,etime=,args= -p "$(echo $1 | tr ' ' ,)" | cut -c1-110 | sed 's/^/  /'
}

all=$(targets)
if [ -z "$all" ]; then
  echo "돌고 있는 학습이 없습니다."
  exit 0
fi

if [ "${1:-}" = "--list" ]; then
  echo "돌고 있는 학습 ($(echo $all | wc -w)개 작업):"
  show "$all"
  exit 0
fi

echo "멈출 작업 ($(echo $all | wc -w)개):"
show "$all"
if [ "${1:-}" != "--yes" ]; then
  read -r -p "멈출까요? [y/N] " answer
  case "$answer" in y|Y|yes) ;; *) echo "그대로 둡니다."; exit 0 ;; esac
fi

for _ in 1 2 3 4 5 6 7 8 9 10; do
  all=$(targets)
  [ -z "$all" ] && break
  kill -TERM $all 2>/dev/null
  sleep 1
done
all=$(targets)
if [ -n "$all" ]; then
  kill -KILL $all 2>/dev/null
  sleep 1
fi
if [ -n "$(targets)" ]; then
  echo "다 멈추지 못했습니다. 남은 작업:"
  show "$(targets)"
  exit 1
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
