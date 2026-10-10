#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# 언어모델 넓히기. run.sh (와 신경망 단계)가 끝난 뒤에 돌린다.
#
#   local/expand_lm.sh --data-root /mnt/e/aihub
#   local/expand_lm.sh --data-root /mnt/e/aihub --plain "내문장.txt"
#
# 지금 언어모델은 음성을 받은 발화의 문장(약 6,800개)만 안다. 라벨에는 음성을 받지
# 않은 문장도 들어 있고, 의료 용어와 직접 모은 문장(--plain)도 더할 수 있다.
# 이것들로 사전과 언어모델을 다시 만들고 그래프만 새로 짠다.
# 음향 모델(소리를 듣는 쪽)은 다시 학습하지 않으므로 몇십 분이면 끝난다.
#
# 평가 문장(data/test/text.word)은 언어모델에서 뺀다. 넣으면 정답을 미리 알려
# 주는 셈이라 점수가 부풀려진다.
#
# 만드는 것
#   data/lang_big_test          새 사전 + 언어모델
#   exp/chain/tree/graph_big    신경망용 그래프. local/transcribe_kaldi.py 는 이게
#                               있으면 이걸 쓴다.
#   exp/tri3/graph_big          신경망이 없을 때만 (tri3 로 평가)
# 끝에 원래 그래프와 점수를 나란히 보여 준다. 나빠졌으면 graph_big 을 지우면
# 원래대로 돌아간다 (rm -rf exp/chain/tree/graph_big).
#
#   0 문장 모으기   1 사전   2 언어모델   3 그래프   4 평가

stage=0
nj=6
data_root=
labels=          # 라벨 폴더나 zip, 여러 개면 따옴표 안에 띄어 쓴다
plain=           # 한 줄에 한 문장인 텍스트 파일, 여러 개면 띄어 쓴다
order=3
decode=true      # false 면 평가(4단계)를 건너뛴다

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ -n "$data_root" ] && [ -z "$labels" ]; then
  for d in $data_root/train/labels $data_root/valid/labels; do
    [ -e $d ] && labels="$labels $d"
  done
fi
if [ -z "$labels$plain" ]; then
  echo "$0: --data-root 나 --labels, --plain 을 알려주세요. 맨 위 사용법 참고." >&2
  exit 1
fi
for f in data/train/text data/test/text.word data/local/morph.bin data/local/dict/lexicon.txt \
    data/lang/phones.txt; do
  [ -f $f ] || { echo "$0: $f 가 없습니다. run.sh 를 먼저 끝까지 돌리세요." >&2; exit 1; }
done

local/check_locale.sh

dir=data/local/lm_big
dict=data/local/dict_big
chain_dir=exp/chain/tdnn1a
tree_dir=exp/chain/tree
if [ -f $chain_dir/final.mdl ]; then
  model=chain graph=$tree_dir/graph_big
else
  model=tri3 graph=exp/tri3/graph_big
fi
nj_of() { local n; n=$(wc -l < $1/spk2utt); echo $(( n < nj ? n : nj )); }
mkdir -p $dir

if [ $stage -le 0 ]; then
  src=
  for l in $labels; do src="$src --labels $l"; done
  for p in $plain; do src="$src --plain $p"; done
  python3 local/aihub.py text $src --exclude data/test/text.word > $dir/sentences.txt
  # 학습 때 배운 조각 규칙 그대로 자른다. 처음 보는 어절도 아는 조각으로 나뉜다.
  python3 local/morph.py apply --no-id --model data/local/morph.bin \
    < $dir/sentences.txt > $dir/sentences.seg
  # 같은 문장은 한 번만 (train_lm.sh 와 같은 이유)
  { cut -d' ' -f2- data/train/text; cat $dir/sentences.seg; } | sort -u > $dir/corpus.txt
  [ -s data/local/terms.seg ] && cat data/local/terms.seg >> $dir/corpus.txt
  echo "$0: 원래 $(cut -d' ' -f2- data/train/text | sort -u | wc -l) 문장 -> $(wc -l < $dir/corpus.txt) 줄"
fi

if [ $stage -le 1 ]; then
  rm -rf $dict $dir/dict_gen
  cat $dir/corpus.txt | python3 local/prepare_dict.py $dir/dict_gen
  mkdir -p $dict
  for f in silence_phones.txt optional_silence.txt nonsilence_phones.txt extra_questions.txt; do
    cp data/local/dict/$f $dict/
  done
  # 음향 모델이 배운 소리로만 발음을 적을 수 있다. 처음 보는 소리가 든 단위는 뺀다.
  awk 'NR == FNR { ok[$1] = 1; next }
       { for (i = 2; i <= NF; i++) if (!($i in ok)) { bad++; next } print }
       END { if (bad) print "모르는 소리가 있어 뺀 단위: " bad > "/dev/stderr" }' \
    <(cat $dict/silence_phones.txt $dict/nonsilence_phones.txt) \
    $dir/dict_gen/lexicon.txt > $dict/lexicon.txt
  # 음소 번호를 원래와 똑같이 맞춘다. 다르면 학습한 모델을 쓸 수 없다.
  utils/prepare_lang.sh --phone-symbol-table data/lang/phones.txt \
    $dict "<UNK>" data/local/lang_big_tmp data/lang_big
  echo "$0: 사전 $(($(wc -l < data/local/dict/lexicon.txt) - 2)) -> $(($(wc -l < $dict/lexicon.txt) - 2)) 단위"
fi

if [ $stage -le 2 ]; then
  python3 utils/lang/make_kn_lm.py -ngram-order $order -text $dir/corpus.txt -lm $dir/lm.arpa
  gzip -f $dir/lm.arpa
  utils/format_lm.sh data/lang_big $dir/lm.arpa.gz $dict/lexicon.txt data/lang_big_test
fi

if [ $stage -le 3 ]; then
  if [ $model = chain ]; then
    utils/lang/check_phones_compatible.sh data/lang_big_test/phones.txt data/lang_chain/phones.txt
    utils/mkgraph.sh --self-loop-scale 1.0 data/lang_big_test $tree_dir $graph
  else
    utils/mkgraph.sh data/lang_big_test exp/tri3 $graph
  fi
fi

if [ $stage -le 4 ] && $decode; then
  if [ $model = chain ]; then
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $(nj_of data/test_hires) --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3/ivectors_test_hires \
      $graph data/test_hires $chain_dir/decode_test_big
    old=$chain_dir/decode_test new=$chain_dir/decode_test_big
    # 녹음 인식이 이 그래프에 맞는 언어모델 가중치를 쓰도록 그래프 옆에 둔다.
    cp $new/scoring_kaldi/best_cer $new/scoring_kaldi/best_wer $graph/ 2>/dev/null || true
  else
    steps/decode_fmllr.sh --nj $(nj_of data/test) --cmd "$decode_cmd" --config conf/decode.config \
      $graph data/test exp/tri3/decode_test_big
    old=exp/tri3/decode_test new=exp/tri3/decode_test_big
  fi
  echo
  echo "== 언어모델 넓히기 전/후 ($model, 낮을수록 좋음) =="
  for d in $old $new; do
    echo "$d"
    cat $d/scoring_kaldi/best_cer $d/scoring_kaldi/best_wer 2>/dev/null | sed 's/^/  /' || true
  done
  echo
  echo "후가 더 낮으면 그대로 둔다 (실제 녹음 인식이 graph_big 을 쓴다)."
  echo "더 높으면: rm -rf $graph"
fi
