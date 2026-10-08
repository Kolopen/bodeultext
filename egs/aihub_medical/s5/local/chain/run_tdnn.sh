#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# TDNN-F chain 모델 (신경망). run.sh 의 tri3 까지 끝난 뒤에 돌린다.
# zeroth_korean 레시피(1a)를 노트북 GPU 한 장(RTX 5070, 8GB)에 맞춰 줄였다.
#
#   local/chain/run_tdnn.sh                 # 처음부터
#   local/chain/run_tdnn.sh --stage 12      # 신경망 학습 단계부터 다시
#   local/chain/run_tdnn.sh --stage 12 --train-stage 37   # 37번째 반복부터 이어서
#
# 학습은 반복마다 모델을 저장하므로 아무 때나 멈췄다가 --train-stage 로 이어
# 할 수 있다. 마지막으로 저장된 번호는 exp/chain/tdnn1a/ 안의 가장 큰 N.mdl 이다.

set -euo pipefail

local/check_locale.sh

stage=0
nj=6
train_set=train
test_sets=test
gmm=tri3
speed_perturb=false
affix=1a

train_stage=-10
use_gpu=true            # false 면 CPU 로 돈다. 아주 느리다(동작 확인용).
num_epochs=4
minibatch=128,64        # GPU 메모리가 모자라면 64,32 로 줄인다.
remove_egs=true
egs_extra=              # 시험용 소량 데이터면 "--num-utts-subset 50"

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if $use_gpu && ! cuda-compiled; then
  echo "$0: Kaldi 가 CUDA 없이 빌드됐습니다. README 의 'Windows(WSL2) + RTX' 절을 보고"
  echo "     src/ 를 --use-cuda 로 다시 빌드하거나, --use-gpu false 로 돌리세요."
  exit 1
fi

nj_of() { local n; n=$(wc -l < $1/spk2utt); echo $(( n < nj ? n : nj )); }

local/nnet3/run_ivector_common.sh --stage $stage --nj $nj --train-set $train_set \
  --test-sets "$test_sets" --gmm $gmm --speed-perturb $speed_perturb

$speed_perturb && train_set=${train_set}_sp
ali_dir=exp/${gmm}_ali_${train_set}
lat_dir=exp/chain/${gmm}_${train_set}_lats
tree_dir=exp/chain/tree
dir=exp/chain/tdnn$affix
lang=data/lang_chain
train_data_dir=data/${train_set}_hires
train_ivector_dir=exp/nnet3/ivectors_${train_set}_hires

if [ $stage -le 8 ]; then
  # 신경망은 소리 하나에 상태 둘짜리 단순한 HMM 을 쓴다.
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl)
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl)
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist > $lang/topo
fi

if [ $stage -le 9 ]; then
  # 정답을 한 줄짜리 정렬 대신 격자(lattice)로 줘서 신경망이 고를 여지를 남긴다.
  steps/align_fmllr_lats.sh --nj $(nj_of data/$train_set) --cmd "$train_cmd" \
    data/$train_set data/lang exp/$gmm $lat_dir
  rm -f $lat_dir/fsts.*.gz
fi

if [ $stage -le 10 ]; then
  rm -rf $tree_dir
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 3500 data/$train_set $lang $ali_dir $tree_dir
fi

if [ $stage -le 11 ]; then
  mkdir -p $dir/configs
  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  xent_regularize=0.1
  learning_rate_factor=$(python3 -c "print(0.5 / $xent_regularize)")
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1024
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/
fi

if [ $stage -le 12 ]; then
  # GPU 가 하나라 작업 수를 1로 고정한다.
  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule '0,0@0.20,0.5@0.50,0' \
    --trainer.srand=0 \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=$num_epochs \
    --trainer.frames-per-iter=1500000 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.num-chunk-per-minibatch=$minibatch \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=140,100,160 \
    --egs.chunk-left-context=0 \
    --egs.chunk-right-context=0 \
    --egs.opts="--frames-overlap-per-eg 0 $egs_extra" \
    --egs.cmd="$train_cmd" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=$use_gpu \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir
fi

if [ $stage -le 13 ]; then
  utils/lang/check_phones_compatible.sh data/lang_test/phones.txt $lang/phones.txt
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $tree_dir $tree_dir/graph
fi

if [ $stage -le 14 ]; then
  for d in $test_sets; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $(nj_of data/${d}_hires) --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3/ivectors_${d}_hires \
      $tree_dir/graph data/${d}_hires $dir/decode_$d
  done
fi

if [ $stage -le 15 ]; then
  # 실제 녹음을 인식할 때 쓰는 묶음. 특징 추출과 i-vector 계산까지 모델 폴더
  # 하나에 들어 있어서, 이 폴더와 그래프만 있으면 다른 컴퓨터(Mac)에서도 인식한다.
  steps/online/nnet3/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3/extractor $dir ${dir}_online
fi

echo
echo "== 신경망 평가 결과 (낮을수록 좋음) =="
for d in $test_sets; do
  cat $dir/decode_$d/scoring_kaldi/best_wer $dir/decode_$d/scoring_kaldi/best_cer 2>/dev/null || true
done
