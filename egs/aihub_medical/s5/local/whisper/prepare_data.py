#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""Kaldi 데이터 폴더(data/train, data/test)를 whisper 학습용 목록(TSV)으로 바꾼다.

  python3 local/whisper/prepare_data.py data/train exp/whisper/train.tsv --max-utts 40000
  python3 local/whisper/prepare_data.py data/test  exp/whisper/test.tsv  --max-utts 500

한 줄에 "발화ID<TAB>wav 경로<TAB>길이(초)<TAB>글" 이다. 글은 Kaldi 와 같은 정규화를
거친 text.word(조각 나누기 전)를 쓴다. 숫자는 한글로 적혀 있으므로 whisper 도 한글로
읽도록 배우고, 숫자로 바꾸는 일은 local/korean_itn.py 가 맡는다.
"""

import argparse
import random
import shlex
import sys
from pathlib import Path


def wav_path(rspec):
    """wav.scp 의 오른쪽에서 원본 파일 경로를 꺼낸다. "sox 경로 -t wav ... |" 꼴도 받는다."""
    rspec = rspec.strip()
    if rspec.endswith("|"):
        parts = shlex.split(rspec[:-1])
        if parts and parts[0] == "sox" and len(parts) > 1:
            return parts[1]
        sys.exit(f"wav.scp 의 이 줄을 읽을 수 없습니다: {rspec}")
    return rspec


def read_map(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, _, rest = line.rstrip("\n").partition(" ")
            if rest:
                out[key] = rest
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, help="Kaldi 데이터 폴더 (wav.scp, text.word, utt2dur)")
    p.add_argument("out", type=Path, help="만들 TSV")
    p.add_argument("--max-utts", type=int, default=0, help="이만큼만 고르게 뽑는다 (0 이면 전부)")
    p.add_argument("--max-sec", type=float, default=30.0, help="whisper 는 30초까지만 듣는다")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    text_file = args.data / "text.word"
    if not text_file.is_file():
        text_file = args.data / "text"
    wavs = read_map(args.data / "wav.scp")
    texts = read_map(text_file)
    durs = {k: float(v) for k, v in read_map(args.data / "utt2dur").items()} \
        if (args.data / "utt2dur").is_file() else {}

    rows = []
    for utt, text in texts.items():
        if utt not in wavs or not text.strip():
            continue
        dur = durs.get(utt, 0.0)
        if dur > args.max_sec:
            continue
        rows.append((utt, wav_path(wavs[utt]), dur, text.strip()))
    rows.sort()
    if args.max_utts and len(rows) > args.max_utts:
        rows = sorted(random.Random(args.seed).sample(rows, args.max_utts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for utt, path, dur, text in rows:
            f.write(f"{utt}\t{path}\t{dur:.2f}\t{text}\n")
    hours = sum(r[2] for r in rows) / 3600
    print(f"{args.out}: 발화 {len(rows)}개, {hours:.1f}시간")


if __name__ == "__main__":
    main()
