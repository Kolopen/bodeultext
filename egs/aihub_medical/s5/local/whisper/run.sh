#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# whisper turbo 를 AI-Hub 의료 음성(의사_1 + 환자_1, Kaldi 와 같은 데이터)으로 추가 학습한다.
# Kaldi 학습(run.sh)이 만든 data/train, data/test 의 발화를 그대로 쓰고, 글은 AI-Hub 라벨
# (--data-root, 기본 /mnt/e/aihub)의 원래 전사에서 문장부호를 살리고 수를 숫자로 바꿔 만든다.
#
#   nohup local/whisper/run.sh > whisper.log 2>&1 &        # 처음부터
#   nohup local/whisper/run.sh --stage 2 > whisper.log 2>&1 &   # 학습을 멈췄다가 이어 하기
#   local/whisper/run.sh --stop-stage 0                      # 환경만 준비
#
#   0 환경 준비 (~/whisper-env: PyTorch, transformers ...)   10~20분
#   1 학습·평가 목록 만들기 (라벨 113만 개를 읽는다)          10~30분
#   2 학습 (GPU, 덧붙임 층 LoRA)                             발화 10만 개에 몇 시간
#   3 faster-whisper 형식으로 내보내기 -> E:\voice-models\fw-aihub, fw-turbo(학습 전 원본)
#   4 AI-Hub 평가 문장으로 원본과 견주기
# 그다음 실제 진료 녹음으로 견주기: local/eval_engines.sh (다음단계.txt [12])
#
# 멈추기는 local/stop_training.sh. 학습은 200 걸음마다 저장하므로 --stage 2 로 다시 하면
# 이어서 한다.

set -euo pipefail

stage=0
stop_stage=4
data_root=/mnt/e/aihub   # Kaldi 때 쓴 AI-Hub 폴더 (train/labels, valid/labels 를 읽는다)
train_data=data/train    # 학습할 발화 (Kaldi 데이터 폴더). 세트를 더 받았으면 data/train_all (다음단계.txt [12-8])
max_utts=100000    # 학습에 쓸 발화 수 (data/train 은 약 18만). 0 이면 전부
max_per_sentence=10   # 같은 문장은 이만큼까지만 쓴다. 한 문장을 수십 명이 읽어서 다 쓰면 외운다
groups=            # 화자 구분만 고를 때. 예: --groups HB (의사만). 비우면 있는 것 전부
pack_sec=28        # 같은 화자의 이어진 발화를 이 길이 안에서 이어 붙인다 (0 이면 안 잇는다)
epochs=1
batch=8            # GPU 메모리가 모자라면(out of memory) 4 로 줄이고 grad_accum 을 4 로
grad_accum=2
train_opts=        # finetune.py 에 더 줄 것. 예: --train-opts "--lr 5e-5 --save-every 100"
base=openai/whisper-large-v3-turbo   # 더 학습할 원본 (HuggingFace 이름 또는 폴더)
name=aihub         # 내보낼 이름: E:\voice-models\fw-이름
env=$HOME/whisper-env
models=/mnt/e/voice-models
dir=exp/whisper

. utils/parse_options.sh

py=$env/bin/python
run() { [ "$stage" -le "$1" ] && [ "$1" -le "$stop_stage" ]; }
# 학습 전 원본도 같은 형식으로 두고 견준다 (turbo 는 fw-turbo, large-v3 는 fw-large-v3)
case "$base" in
  *large-v3-turbo) orig=$models/fw-turbo ;;
  *) orig=$models/fw-$(basename "$base" | sed 's/^whisper-//') ;;
esac
trained=$models/fw-$name

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
  for d in "$data_root/train/labels" "$data_root/valid/labels"; do
    [ -e "$d" ] || { echo "AI-Hub 라벨이 없습니다: $d (--data-root 로 알려 주세요)" >&2; exit 1; }
  done
  mkdir -p $dir
  # 평가 문장(학습에 없던 문장)에서 학습 중 확인용 300, 마지막 평가용 500 을 고른다
  "$py" local/whisper/prepare_data.py data/test $dir/all_test.tsv --labels "$data_root/valid/labels"
  "$py" - "$dir" data/test/text <<'EOF'
import random, sys
d, kaldi_text = sys.argv[1], sys.argv[2]
rows = open(f"{d}/all_test.tsv", encoding="utf-8").read().splitlines()
random.Random(1).shuffle(rows)
valid, test = rows[:300], rows[300:800]
open(f"{d}/valid.tsv", "w", encoding="utf-8").writelines(r + "\n" for r in valid)
open(f"{d}/test.tsv", "w", encoding="utf-8").writelines(r + "\n" for r in test)
# 고른 평가 문장(Kaldi 꼴 글)을 따로 적어 둔다. 학습 목록에서 같은 문장을 뺄 때 쓴다
chosen = {r.split("\t", 1)[0] for r in valid + test}
with open(kaldi_text, encoding="utf-8") as f, open(f"{d}/eval_text", "w", encoding="utf-8") as out:
    out.writelines(line for line in f if line.split(maxsplit=1)[0] in chosen)
print(f"확인용 {len(valid)}개, 평가용 {len(test)}개")
if len(rows) < 800:
    print("  평가 문장이 적습니다. data/test 를 확인하세요.")
EOF
  # 평가 문장과 같은 문장은 다른 사람이 읽은 것도 학습에서 뺀다(외워서 맞히지 않게)
  "$py" local/whisper/prepare_data.py $train_data $dir/train.tsv --labels "$data_root/train/labels" \
    --max-utts $max_utts --max-per-sentence $max_per_sentence --pack-sec $pack_sec \
    --holdout $dir/eval_text ${groups:+--groups $groups}
fi

if run 2; then
  "$py" local/whisper/finetune.py --base $base --train $dir/train.tsv --valid $dir/valid.tsv --out $dir/lora \
    --epochs $epochs --batch $batch --grad-accum $grad_accum $train_opts
fi

if run 3; then
  mkdir -p $models
  [ -f $orig/model.bin ] || "$py" local/whisper/export.py --base $base --out $orig
  "$py" local/whisper/export.py --lora $dir/lora/final --out $trained
fi

if run 4; then
  # faster-whisper 가 GPU 를 쓰려면 PyTorch 와 같이 깔린 cuBLAS, cuDNN 을 찾아야 한다
  libs=$("$py" -c 'import nvidia.cublas.lib as a, nvidia.cudnn.lib as b; print(a.__path__[0] + ":" + b.__path__[0])' 2>/dev/null || true)
  [ -n "$libs" ] && export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "== AI-Hub 평가 문장 (학습에 없던 문장. 글자 오류율은 낮을수록 좋음)"
  "$py" local/whisper/eval_test.py $dir/test.tsv $orig $trained | tee $dir/eval_test.txt
  echo
  echo "실제 진료 녹음으로 견주기 ($(basename $trained) 가 자동으로 들어간다):"
  echo "  local/eval_engines.sh /mnt/e/녹음/0001.m4a /mnt/e/녹음/0002.m4a /mnt/e/녹음/0003.m4a"
fi
