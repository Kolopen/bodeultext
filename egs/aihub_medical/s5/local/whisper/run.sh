#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# whisper turbo 를 AI-Hub 의료 음성(의사_1 + 환자_1, Kaldi 와 같은 데이터)으로 추가 학습한다.
# Kaldi 학습(run.sh)이 만든 data/train, data/test 를 그대로 쓴다.
#
#   nohup local/whisper/run.sh > whisper.log 2>&1 &        # 처음부터
#   nohup local/whisper/run.sh --stage 2 > whisper.log 2>&1 &   # 학습을 멈췄다가 이어 하기
#   local/whisper/run.sh --stop-stage 0                      # 환경만 준비
#
#   0 환경 준비 (~/whisper-env: PyTorch, transformers ...)   10~20분
#   1 학습·평가 목록 만들기                                   몇 분
#   2 학습 (GPU, 덧붙임 층 LoRA)                             발화 4만 개에 몇 시간
#   3 faster-whisper 형식으로 내보내기 -> E:\voice-models\fw-aihub, fw-turbo(원본)
#   4 AI-Hub 평가 문장으로 원본과 견주기
# 그다음 실제 진료 녹음으로 견주기: local/eval_engines.sh (다음단계.txt [12])
#
# 멈추기는 local/stop_training.sh. 학습은 200 걸음마다 저장하므로 --stage 2 로 다시 하면
# 이어서 한다.

set -euo pipefail

stage=0
stop_stage=4
max_utts=40000     # 학습에 쓸 발화 수. 전부(18만)면 한 바퀴에 하루 가까이 걸린다
epochs=1
batch=8            # GPU 메모리가 모자라면(out of memory) 4 로 줄이고 grad_accum 을 4 로
grad_accum=2
train_opts=        # finetune.py 에 더 줄 것. 예: --train-opts "--lr 5e-5 --save-every 100"
base=openai/whisper-large-v3-turbo   # 더 학습할 원본 (HuggingFace 이름 또는 폴더)
env=$HOME/whisper-env
models=/mnt/e/voice-models
dir=exp/whisper

. utils/parse_options.sh

py=$env/bin/python
run() { [ "$stage" -le "$1" ] && [ "$1" -le "$stop_stage" ]; }

if run 0; then
  if [ ! -x "$py" ]; then
    python3 -m venv "$env" || {
      echo "가상환경을 못 만들었습니다. 먼저: sudo apt install -y python3-venv" >&2; exit 1; }
  fi
  "$env/bin/pip" install -q --upgrade pip
  # RTX 50 시리즈는 CUDA 12.8 용 PyTorch 가 필요하다.
  "$env/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu128
  "$env/bin/pip" install -q "transformers>=4.45,<5" "peft>=0.13" accelerate soundfile scipy \
    ctranslate2 faster-whisper huggingface_hub
  "$py" -c "
import torch
assert torch.cuda.is_available(), 'GPU 를 못 찾았습니다 (nvidia-smi 확인)'
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch', torch.__version__)
x = torch.randn(256, 256, device='cuda'); print('GPU 계산 확인:', float((x @ x).sum()) == float((x @ x).sum()))
"
fi

if run 1; then
  mkdir -p $dir
  "$py" local/whisper/prepare_data.py data/train $dir/train.tsv --max-utts $max_utts
  # 평가 문장(학습에 없던 문장)을 둘로 나눈다: 학습 중 확인용 300, 마지막 평가용 500
  "$py" local/whisper/prepare_data.py data/test $dir/all_test.tsv
  "$py" - "$dir" <<'EOF'
import random, sys
d = sys.argv[1]
rows = open(f"{d}/all_test.tsv", encoding="utf-8").read().splitlines()
random.Random(1).shuffle(rows)
open(f"{d}/valid.tsv", "w", encoding="utf-8").writelines(r + "\n" for r in rows[:300])
open(f"{d}/test.tsv", "w", encoding="utf-8").writelines(r + "\n" for r in rows[300:800])
print(f"확인용 {len(rows[:300])}개, 평가용 {len(rows[300:800])}개")
EOF
fi

if run 2; then
  "$py" local/whisper/finetune.py --base $base --train $dir/train.tsv --valid $dir/valid.tsv --out $dir/lora \
    --epochs $epochs --batch $batch --grad-accum $grad_accum $train_opts
fi

if run 3; then
  mkdir -p $models
  [ -f $models/fw-turbo/model.bin ] || "$py" local/whisper/export.py --base $base --out $models/fw-turbo
  "$py" local/whisper/export.py --lora $dir/lora/final --out $models/fw-aihub
fi

if run 4; then
  # faster-whisper 가 GPU 를 쓰려면 PyTorch 와 같이 깔린 cuBLAS, cuDNN 을 찾아야 한다
  libs=$("$py" -c 'import nvidia.cublas.lib as a, nvidia.cudnn.lib as b; print(a.__path__[0] + ":" + b.__path__[0])' 2>/dev/null || true)
  [ -n "$libs" ] && export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  others=()
  for d in $models/fw-*; do
    case "$d" in */fw-turbo|*/fw-aihub) ;; *) [ -f "$d/model.bin" ] && others+=("$d") ;; esac
  done
  echo "== AI-Hub 평가 문장 500개 (낮을수록 좋음)"
  "$py" local/whisper/eval_test.py $dir/test.tsv $models/fw-turbo $models/fw-aihub "${others[@]}" \
    | tee $dir/eval_test.txt
  echo
  echo "실제 진료 녹음으로 견주기 (fw-aihub 가 자동으로 들어간다):"
  echo "  local/eval_engines.sh /mnt/e/녹음/0001.m4a /mnt/e/녹음/0002.m4a /mnt/e/녹음/0003.m4a"
fi
