#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""faster-whisper 모델들을 AI-Hub 평가 문장으로 잰다 (Kaldi 평가와 같은 문장).

  python local/whisper/eval_test.py exp/whisper/test.tsv /mnt/e/voice-models/fw-turbo /mnt/e/voice-models/fw-aihub

학습에 없던 문장이라 의료 낭독체에서 얼마나 나아졌는지를 본다. 실제 진료 녹음에서의
성능은 local/eval_engines.sh 로 따로 잰다.
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from finetune import cer, load_audio, read_tsv  # noqa: E402
from compare_asr import hangul  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test", type=Path)
    p.add_argument("models", nargs="+", type=Path)
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--max-utts", type=int, default=500)
    args = p.parse_args()

    from faster_whisper import WhisperModel

    def run(model, audio):
        segments, _ = model.transcribe(audio, language="ko", beam_size=5, condition_on_previous_text=False)
        return " ".join(s.text for s in segments)

    rows = read_tsv(args.test)[:args.max_utts]
    device = args.device
    for path in args.models:
        model = WhisperModel(str(path), device=device, compute_type="default")
        try:
            run(model, load_audio(rows[0][1]))
        except RuntimeError as error:  # GPU 라이브러리(cuDNN, cuBLAS)를 못 찾으면 CPU 로
            if device == "cpu":
                raise
            print(f"GPU 로 못 돌려 CPU 로 합니다 (느림): {str(error).splitlines()[0]}", flush=True)
            device = "cpu"
            model = WhisperModel(str(path), device=device, compute_type="int8")
        refs, hyps, started = [], [], time.time()
        for _, wav, _, text in rows:
            # 문장부호·띄어쓰기는 빼고, 숫자는 한글로 읽어서 견준다(compare_asr.py 와 같게)
            hyps.append("".join(hangul(run(model, load_audio(wav)))))
            refs.append("".join(hangul(text)))
        print(f"{path.name:<24} 글자 오류율 {cer(refs, hyps):6.2f}%  ({len(rows)}문장, "
              f"{time.time() - started:.0f}초)", flush=True)


if __name__ == "__main__":
    main()
