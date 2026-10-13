#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# 진료 녹음 하나 -> 받아 적기(학습한 Kaldi 모델) -> 리포트(-AI 의 voice-analyze)
#
# whisper turbo 모델(VOICE_WHISPER)이 있으면 두 엔진을 같이 쓴다. 본문은 whisper,
# 비교는 Kaldi 로 돌리고, 두 엔진의 숫자·용어가 엇갈리는 자리를 리포트에 표시한다.
# REPORT_ENGINE=kaldi 로 주면 Kaldi 만 쓴다.
#
#   local/report_audio.sh 녹음.m4a [진료과] [진료일]
#   local/report_audio.sh /mnt/e/녹음/0102.m4a 내과 2026-10-02
#
# 진료과는 -AI 의 용어 사전 이름이다(공통 내과 신경과 안과 정형외과). 기본 내과.
# 진료일은 의사가 말한 "10월 20일"의 연도를 정하는 데 쓴다. 기본 오늘.
#
# 결과는 녹음 옆에 녹음 이름으로 폴더를 만들어 넣는다.
#   /mnt/e/녹음/0102.m4a -> /mnt/e/녹음/0102/
#     transcript.txt   받아 적은 글 ([시각] 화자: 내용)
#     transcript.json  voice-analyze 에 넣는 형식
#     report.txt       리포트 ([검사 수치] [진단·소견] ... 리포트 초안)
#     report.json      같은 내용의 JSON
#
# 화자분리 모델이 VOICE_MODELS(기본 ~/voice-models)에 있으면 의사/환자를 목소리로
# 나눈다. 없으면 나누지 않고, 리포트 단계가 문장 말투로 의사/환자를 가른다.
# 받는 법은 다음단계.txt 의 [6].
#
# 바꿀 수 있는 것 (환경 변수)
#   VOICE_MODELS    화자분리 모델 폴더        기본 ~/voice-models
#   VOICE_SPEAKERS  화자 수 (모르면 -1)       기본 2
#   VOICE_MANAGER   매니저 성문(.json)        기본 없음
#   VOICE_OUT       결과 폴더                 기본 녹음 옆
#   KALDI_MODEL     모델 폴더                 기본 exp/chain/tdnn1a_online
#   KALDI_GRAPH     그래프 폴더               기본 graph_big 이 있으면 그것, 없으면 graph
#   KALDI_NJ        동시 인식 작업 수         기본 4
#   KALDI_OPTS      transcribe_kaldi.py 에 그대로 넘길 옵션  예: "--lmwt 12 --pause 0.3"
#   VOICE_WHISPER   whisper 폴더              기본 /mnt/e/voice-models/sherpa-onnx-whisper-turbo
#                   faster-whisper 폴더(model.bin 이 든 것)도 된다. 예: 직접 학습한
#                   /mnt/e/voice-models/fw-aihub (다음단계.txt [12])
#   REPORT_ENGINE   auto(있으면 둘 다) / kaldi / dual   기본 auto

set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '5,41p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

audio=$(realpath "$1")
department=${2:-내과}
consult_date=${3:-$(date +%F)}
[ -f "$audio" ] || { echo "녹음 파일이 없습니다: $1" >&2; exit 1; }

models=${VOICE_MODELS:-$HOME/voice-models}
speakers=${VOICE_SPEAKERS:-2}
name=$(basename "${audio%.*}")
out=${VOICE_OUT:-$(dirname "$audio")/$name}
mkdir -p "$out"

cd "$(dirname "$0")/.."
export PYTHONPATH="$HOME/-AI/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8

opts=()
[ -n "${KALDI_MODEL:-}" ] && opts+=(--model "$KALDI_MODEL")
[ -n "${KALDI_GRAPH:-}" ] && opts+=(--graph "$KALDI_GRAPH")
segmentation=$models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx
embedding=$models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
if [ -f "$segmentation" ] && [ -f "$embedding" ]; then
  opts+=(--segmentation "$segmentation" --embedding "$embedding" --speakers "$speakers")
  [ -n "${VOICE_MANAGER:-}" ] && opts+=(--manager "$(realpath "$VOICE_MANAGER")")
  echo "화자분리: 사용 (화자 $speakers 명)"
elif [ -f "$models/silero_vad.onnx" ]; then
  opts+=(--vad "$models/silero_vad.onnx")
  echo "화자분리: 없음 (말소리 구간만 자름)"
else
  echo "화자분리: 없음 ($models 에 모델이 없음. 다음단계.txt [6] 참고)"
fi
if [ -n "${VOICE_MANAGER:-}" ] && ! [ -f "$segmentation" ]; then
  echo "VOICE_MANAGER 는 화자분리 모델(다음단계.txt [6-2])이 있어야 씁니다. 이번에는 무시합니다." >&2
fi

whisper=${VOICE_WHISPER:-/mnt/e/voice-models/sherpa-onnx-whisper-turbo}
wengine=()
if [ -f "$whisper/model.bin" ]; then
  wengine=(--engine faster-whisper --model "$whisper")
  wname="faster-whisper: $(basename "$whisper")"
elif [ -f "$whisper/turbo-encoder.int8.onnx" ]; then
  wengine=(--engine whisper --encoder "$whisper/turbo-encoder.int8.onnx"
           --decoder "$whisper/turbo-decoder.int8.onnx" --tokens "$whisper/turbo-tokens.txt")
  wname="whisper turbo"
fi
engine=${REPORT_ENGINE:-auto}
if [ "$engine" = auto ]; then
  [ ${#wengine[@]} -gt 0 ] && engine=dual || engine=kaldi
fi
if [ "$engine" = dual ]; then
  [ ${#wengine[@]} -gt 0 ] || { echo "whisper 모델이 없습니다: $whisper (다음단계.txt [11], [12])" >&2; exit 1; }
  if [ "${wengine[1]}" = faster-whisper ] && ! python3 -c "import faster_whisper" 2>/dev/null; then
    echo "faster-whisper 가 없습니다 (다음단계.txt [12-1])." >&2; exit 1
  fi
  [ -f "$segmentation" ] || [ -f "$models/silero_vad.onnx" ] || {
    echo "whisper 를 쓰려면 화자분리 모델(다음단계.txt [6-2])이 있어야 합니다." >&2; exit 1; }
fi

echo "결과 폴더: $out"
echo "[1/3] 받아 적기 (Kaldi)"
python3 local/transcribe_kaldi.py "$audio" --out "$out/transcript.json" \
  --nj "${KALDI_NJ:-4}" "${opts[@]}" ${KALDI_OPTS:-}

main=$out/transcript.json
compare=()
if [ "$engine" = dual ]; then
  echo "[2/3] 받아 적기 ($wname)"
  wopts=(--vad "$models/silero_vad.onnx")
  [ -f "$segmentation" ] && wopts=(--segmentation "$segmentation" --embedding "$embedding" --speakers "$speakers")
  [ -n "${VOICE_MANAGER:-}" ] && [ -f "$segmentation" ] && wopts+=(--manager "$(realpath "$VOICE_MANAGER")")
  python3 -m voice_ai.transcribe "$audio" "${wengine[@]}" "${wopts[@]}" --threads "$(nproc)" \
    --out "$out/whisper.json"
  main=$out/whisper.json
  compare=(--compare "$out/transcript.json")
else
  echo "[2/3] whisper 안 씀 (Kaldi 만)"
fi

echo "[3/3] 리포트 (voice-analyze)"
python3 -m voice_ai.analyze "$main" "${compare[@]}" \
  --department "$department" --date "$consult_date" | tee "$out/report.txt"
python3 -m voice_ai.analyze "$main" "${compare[@]}" \
  --department "$department" --date "$consult_date" --json > "$out/report.json"

echo
echo "받아 적은 글 전체: $out/transcript.txt"
echo "전체 대화를 화자와 함께 보려면:"
echo "  PYTHONPATH=~/-AI/src python3 -m voice_ai.analyze \"$main\" --department $department --full"
