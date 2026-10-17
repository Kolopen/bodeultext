#!/usr/bin/env bash
# Copyright 2026  Kolopen
# Apache 2.0
#
# 같은 녹음을 여러 엔진으로 받아 적고 정답(클로바노트)과 견준다.
#
#   local/eval_engines.sh /mnt/e/녹음/0102.m4a /mnt/e/녹음/0103.m4a /mnt/e/녹음/0104.m4a
#
# 녹음마다 같은 이름의 정답 텍스트를 옆에 둔다.
#   /mnt/e/녹음/0102_대본.txt   실제로 한 말(원본 대본). 있으면 이것이 정답이고,
#                               클로바노트(0102.txt)도 한 엔진으로 함께 잰다.
#   /mnt/e/녹음/0102.txt        클로바노트에서 텍스트로 내보낸 것. 대본이 없으면 이것이 정답.
#
# 결과
#   /mnt/e/녹음/0102/kaldi.json, whisper.json, sense.json   엔진별 받아 적은 글
#   /mnt/e/녹음/엔진비교.txt                                  견준 표
# 이미 만든 json 은 건너뛴다(다시 하려면 지운다).
#
# 모델 위치 (환경 변수로 바꾼다)
#   VOICE_MODELS    화자분리 모델     기본 ~/voice-models
#   VOICE_WHISPER   whisper turbo     기본 /mnt/e/voice-models/sherpa-onnx-whisper-turbo
#   VOICE_SENSE     SenseVoice (선택) 기본 /mnt/e/voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
#   VOICE_SPEAKERS  화자 수           기본 2
#   VOICE_FW        faster-whisper 모델들 (한국어 파인튜닝 등). "이름=폴더" 를 띄어 쓴다.
#                   기본: /mnt/e/voice-models/fw-* 폴더를 모두 (이름은 fw- 뒤)
#                   결과는 fw-이름.json 으로 남는다. 다음단계.txt [12]
#   VOICE_FW_OPTS   faster-whisper 로 받아 적을 때 -AI 에 더 줄 옵션. 예: "--max-chunk 30"
#   VOICE_FW_TAG    그때 결과 이름 뒤에 붙일 말. 예: -long -> fw-이름-long.json (기존 것과 견준다)

set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '5,28p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

models=${VOICE_MODELS:-$HOME/voice-models}
whisper=${VOICE_WHISPER:-/mnt/e/voice-models/sherpa-onnx-whisper-turbo}
sense=${VOICE_SENSE:-/mnt/e/voice-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17}
speakers=${VOICE_SPEAKERS:-2}
threads=$(nproc)
segmentation=$models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx
embedding=$models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

for f in "$segmentation" "$embedding"; do
  [ -f "$f" ] || { echo "화자분리 모델이 없습니다: $f (다음단계.txt [6-2])" >&2; exit 1; }
done
[ -f "$whisper/turbo-encoder.int8.onnx" ] || echo "whisper turbo 가 없어 건너뜁니다: $whisper (다음단계.txt [11])" >&2

audios=()
for a in "$@"; do audios+=("$(realpath "$a")"); done
cd "$(dirname "$0")/.."
export PYTHONPATH="$HOME/-AI/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8
diar=(--segmentation "$segmentation" --embedding "$embedding" --speakers "$speakers")

# faster-whisper 모델 목록: "이름=폴더 이름=폴더 ..."
fw=${VOICE_FW:-}
fw_opts=${VOICE_FW_OPTS:-}
fw_tag=${VOICE_FW_TAG:-}
if [ -z "$fw" ]; then
  for d in /mnt/e/voice-models/fw-*; do
    [ -f "$d/model.bin" ] && fw="$fw ${d##*/fw-}=$d"
  done
fi
if [ -n "$fw" ] && ! python3 -c "import faster_whisper" 2>/dev/null; then
  echo "faster-whisper 가 없어 그 모델들은 건너뜁니다 (다음단계.txt [12-1])." >&2
  fw=
fi

for audio in "${audios[@]}"; do
  out=${audio%.*}
  mkdir -p "$out"
  echo "== $(basename "$audio")"
  [ -f "${audio%.*}.txt" ] || [ -f "${audio%.*}_대본.txt" ] || echo "  정답(${audio%.*}_대본.txt 나 .txt)이 없습니다. 표에서 빠집니다." >&2
  if [ ! -f "$out/kaldi.json" ]; then
    echo "  [Kaldi]"
    python3 local/transcribe_kaldi.py "$audio" --out "$out/kaldi.json" "${diar[@]}" | sed 's/^/    /'
  fi
  if [ -f "$whisper/turbo-encoder.int8.onnx" ] && [ ! -f "$out/whisper.json" ]; then
    echo "  [whisper turbo]"
    python3 -m voice_ai.transcribe "$audio" --engine whisper \
      --encoder "$whisper/turbo-encoder.int8.onnx" --decoder "$whisper/turbo-decoder.int8.onnx" \
      --tokens "$whisper/turbo-tokens.txt" "${diar[@]}" --threads "$threads" \
      --out "$out/whisper.json" | sed 's/^/    /'
  fi
  for item in $fw; do
    name=${item%%=*}$fw_tag dir=${item#*=}
    [ -f "$out/fw-$name.json" ] && continue
    echo "  [faster-whisper: $name${fw_opts:+ ($fw_opts)}]"
    python3 -m voice_ai.transcribe "$audio" --engine faster-whisper --model "$dir" \
      "${diar[@]}" --threads "$threads" $fw_opts --out "$out/fw-$name.json" | sed 's/^/    /'
  done
  if [ -f "$sense/model.int8.onnx" ] && [ ! -f "$out/sense.json" ]; then
    echo "  [SenseVoice]"
    python3 -m voice_ai.transcribe "$audio" --engine sensevoice \
      --model "$sense/model.int8.onnx" --tokens "$sense/tokens.txt" "${diar[@]}" --threads "$threads" \
      --out "$out/sense.json" | sed 's/^/    /'
  fi
done

table=$(dirname "${audios[0]}")/엔진비교.txt
echo
python3 local/compare_asr.py "${audios[@]}" | tee "$table"
echo "표: $table"
