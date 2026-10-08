#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# 신경망 학습에 쓰는 고해상도 특징(40차원 MFCC)과 화자 특성 벡터(i-vector)를 만든다.
# 전부 CPU 작업이다.

set -euo pipefail

stage=0
nj=6
train_set=train
test_sets=test
gmm=tri3
speed_perturb=false   # true 면 데이터를 0.9/1.0/1.1 배속으로 3배 불린다. 정확도는 오르고 시간도 3배.

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

nj_of() { local n; n=$(wc -l < $1/spk2utt); echo $(( n < nj ? n : nj )); }

if $speed_perturb; then
  if [ $stage -le 1 ] && [ ! -f data/${train_set}_sp/feats.scp ]; then
    utils/data/perturb_data_dir_speed_3way.sh data/$train_set data/${train_set}_sp
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/${train_set}_sp exp/make_mfcc/${train_set}_sp mfcc
    steps/compute_cmvn_stats.sh data/${train_set}_sp exp/make_mfcc/${train_set}_sp mfcc
    utils/fix_data_dir.sh data/${train_set}_sp
  fi
  train_set=${train_set}_sp
fi

if [ $stage -le 2 ] && [ ! -f exp/${gmm}_ali_${train_set}/ali.1.gz ]; then
  # 신경망의 정답 상태열과 트리는 tri3 정렬에서 나온다.
  steps/align_fmllr.sh --nj $(nj_of data/$train_set) --cmd "$train_cmd" \
    data/$train_set data/lang exp/$gmm exp/${gmm}_ali_${train_set}
fi

if [ $stage -le 3 ]; then
  for d in $train_set $test_sets; do
    [ -f data/${d}_hires/feats.scp ] && continue
    utils/copy_data_dir.sh --validate-opts "--non-print" data/$d data/${d}_hires
    steps/make_mfcc.sh --nj $(nj_of data/${d}_hires) --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${d}_hires exp/make_mfcc/${d}_hires mfcc
    steps/compute_cmvn_stats.sh data/${d}_hires exp/make_mfcc/${d}_hires mfcc
    utils/fix_data_dir.sh data/${d}_hires
  done
fi

if [ $stage -le 4 ]; then
  total=$(wc -l < data/${train_set}_hires/utt2spk)
  n=$(( total < 30000 ? total : 30000 ))
  utils/subset_data_dir.sh data/${train_set}_hires $n data/${train_set}_subset_hires
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" --max-utts 30000 --subsample 2 \
    data/${train_set}_subset_hires exp/nnet3/pca_transform
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $(nj_of data/${train_set}_subset_hires) \
    --num-frames 700000 data/${train_set}_subset_hires 512 exp/nnet3/pca_transform exp/nnet3/diag_ubm
fi

if [ $stage -le 6 ]; then
  # 메모리를 아끼려고 스레드·프로세스 수를 낮춘다 (기본값은 4 x 4).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 2 \
    --num-threads 2 --num-processes 2 \
    data/${train_set}_hires exp/nnet3/diag_ubm exp/nnet3/extractor
fi

if [ $stage -le 7 ]; then
  # 학습 때는 발화 두 개씩 묶어 화자로 본다. 실제 인식은 녹음 하나로 하므로
  # 화자 정보가 적은 상황에 익숙해지게 한다.
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_hires data/${train_set}_hires_max2
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $(nj_of data/${train_set}_hires_max2) \
    data/${train_set}_hires_max2 exp/nnet3/extractor exp/nnet3/ivectors_${train_set}_hires
  for d in $test_sets; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $(nj_of data/${d}_hires) \
      data/${d}_hires exp/nnet3/extractor exp/nnet3/ivectors_${d}_hires
  done
fi
