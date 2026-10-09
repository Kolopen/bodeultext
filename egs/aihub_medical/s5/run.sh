#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# AI-Hub '의료진 및 환자 음성'으로 한국어 음성인식 모델을 학습한다.
# 0~9 단계(GMM)는 CPU 로, 10 단계(신경망)는 NVIDIA GPU 로 돈다.
# Windows 는 WSL2 안에서 돌린다(README 참고).
#
#   ./run.sh --train-labels ~/aihub/train/labels \
#            --train-audio  ~/aihub/train/audio \
#            --test-labels  ~/aihub/valid/labels \
#            --test-audio   ~/aihub/valid/audio \
#            --terms ~/-AI/src/voice_ai/data/terms
#
# 중간에 멈췄으면 --stage N 으로 그 단계부터 다시 시작한다.
#
#   0 데이터 폴더   1 조각 나누기   2 발음사전   3 언어모델   4 특징 추출
#   5 mono   6 tri1   7 tri2 (LDA+MLLT)   8 tri3 (SAT)   9 tri3 평가
#   10 신경망 (local/chain/run_tdnn.sh, GPU)

stage=0
nj=6                # 동시에 돌릴 CPU 작업 수. 메모리가 모자라면 줄인다.
train_labels=
train_audio=
test_labels=
test_audio=
terms=              # -AI 의 data/terms 폴더. 비워 두면 용어 없이 간다.
test_utts=3000      # 평가에 쓸 발화 수. 전부 쓰면 디코딩이 오래 걸린다.
decode_all=false    # true 면 mono/tri1/tri2 도 하나하나 평가한다.
chain=true          # false 면 tri3 에서 멈춘다.

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

local/check_locale.sh

if [ $stage -le 0 ]; then
  for v in train_labels train_audio test_labels test_audio; do
    if [ -z "${!v}" ]; then
      echo "$0: --${v//_/-} 를 알려주세요. 맨 위 사용법 참고." >&2
      exit 1
    fi
  done
  python3 local/aihub.py prep --labels $train_labels --audio $train_audio --out data/train
  # 평가는 학습에 없던 문장으로만 한다. 같은 문장을 외워서 맞히는 점수를 피한다.
  python3 local/aihub.py prep --labels $test_labels --audio $test_audio --out data/test \
    --max-utts $test_utts --unseen-from data/train/text
  for d in train test; do
    cp data/$d/text data/$d/text.word
    utils/fix_data_dir.sh data/$d
  done
fi

if [ $stage -le 1 ]; then
  mkdir -p data/local
  extra=
  if [ -n "$terms" ]; then
    python3 local/extract_terms.py $terms > data/local/terms.txt
    extra=data/local/terms.txt
  fi
  # 조각 규칙은 학습 전사로만 배운다. 평가 전사를 보면 점수가 부풀려진다.
  python3 local/morph.py train --model data/local/morph.bin data/train/text.word $extra
  for d in train test; do
    python3 local/morph.py apply --model data/local/morph.bin < data/$d/text.word > data/$d/text
  done
  if [ -n "$extra" ]; then
    python3 local/morph.py apply --no-id --model data/local/morph.bin \
      < $extra > data/local/terms.seg
  else
    : > data/local/terms.seg
  fi
fi

if [ $stage -le 2 ]; then
  cut -d' ' -f2- data/train/text | cat - data/local/terms.seg \
    | python3 local/prepare_dict.py data/local/dict
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang_tmp data/lang
fi

if [ $stage -le 3 ]; then
  local/train_lm.sh data/train/text data/local/terms.seg data/local/lm
  utils/format_lm.sh data/lang data/local/lm/lm.arpa.gz \
    data/local/dict/lexicon.txt data/lang_test
fi

if [ $stage -le 4 ]; then
  for d in train test; do
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/$d exp/make_mfcc/$d mfcc
    steps/compute_cmvn_stats.sh data/$d exp/make_mfcc/$d mfcc
    utils/fix_data_dir.sh data/$d
  done
  # 처음 단계는 적은 데이터로 빨리 돈다. 발화가 짧은 낭독체라 개수를 넉넉히 잡는다.
  total=$(wc -l < data/train/utt2spk)
  subset() {  # 데이터가 적으면 있는 만큼만 쓴다.
    local n=$(( $2 < total ? $2 : total ))
    utils/subset_data_dir.sh $1 data/train $n data/$3
  }
  subset --shortest 3000 train_3kshort
  subset "" 15000 train_15k
  subset "" 40000 train_40k
fi

# 화자 수보다 작업을 많이 나눌 수 없다.
nj_of() { local n; n=$(wc -l < $1/spk2utt); echo $(( n < nj ? n : nj )); }

decode() {
  local model=$1 script=${2:-steps/decode.sh}
  utils/mkgraph.sh data/lang_test exp/$model exp/$model/graph
  $script --nj $(nj_of data/test) --cmd "$decode_cmd" --config conf/decode.config \
    exp/$model/graph data/test exp/$model/decode_test
}

if [ $stage -le 5 ]; then
  steps/train_mono.sh --boost-silence 1.25 --nj $(nj_of data/train_3kshort) --cmd "$train_cmd" \
    data/train_3kshort data/lang exp/mono
  $decode_all && decode mono
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj $(nj_of data/train_15k) --cmd "$train_cmd" \
    data/train_15k data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_15k data/lang exp/mono_ali exp/tri1
  $decode_all && decode tri1
fi

if [ $stage -le 7 ]; then
  steps/align_si.sh --nj $(nj_of data/train_40k) --cmd "$train_cmd" \
    data/train_40k data/lang exp/tri1 exp/tri1_ali
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    2500 15000 data/train_40k data/lang exp/tri1_ali exp/tri2
  $decode_all && decode tri2
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $(nj_of data/train) --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali
  steps/train_sat.sh --cmd "$train_cmd" \
    4200 40000 data/train data/lang exp/tri2_ali exp/tri3
fi

if [ $stage -le 9 ]; then
  decode tri3 steps/decode_fmllr.sh
  echo
  echo "== 평가 결과 (낮을수록 좋음) =="
  grep -h WER exp/*/decode_test/scoring_kaldi/best_wer exp/*/decode_test/scoring_kaldi/best_cer \
    2>/dev/null || true
  echo
  echo "best_cer 가 글자 오류율이다. 한국어는 이쪽으로 비교한다."
  echo "주의: 이 데이터는 같은 문장을 여러 사람이 읽은 낭독체라, 평가 문장이 학습에도"
  echo "      있을 수 있다. 실제 진료 녹음에서의 성능은 따로 재야 한다(README 참고)."
fi

if [ $stage -le 10 ] && $chain; then
  if ! cuda-compiled; then
    echo "$0: Kaldi 가 CUDA 없이 빌드돼서 신경망 단계는 건너뜁니다."
    echo "     GPU 컴퓨터에서 'local/chain/run_tdnn.sh' 를 따로 돌리세요."
    exit 0
  fi
  local/chain/run_tdnn.sh --nj $nj
fi
