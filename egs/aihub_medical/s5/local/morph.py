#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""어절을 형태소 비슷한 조각으로 나눈다 (Morfessor).

한국어는 어절에 조사·어미가 붙어 같은 말이 수없이 많은 모양으로 나온다.
어절을 그대로 단어로 쓰면 학습 때 못 본 어절은 인식할 수 없다. 조각으로
나누면 "얼굴이"를 못 봤어도 "얼굴"과 "+이"로 알아듣는다.

어절 안쪽 조각에는 앞에 "+"를 붙인다. 인식 결과에서 " +"만 지우면 원래
띄어쓰기가 돌아온다.

  얼굴이 이상해지는 거  ->  얼굴 +이 이상 +해지 +는 거

  python3 local/morph.py train --model data/local/morph.bin data/train/text [단어목록 ...]
  python3 local/morph.py apply --model data/local/morph.bin < text > text.seg
"""

import argparse
import collections
import math
import sys

import morfessor


def read_kaldi_text(path):
    """Kaldi text(발화ID 전사)와 단어 목록(한 줄에 하나) 둘 다 받는다."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            yield parts[1:] if path.endswith("text") else parts


def cmd_train(args):
    counts = collections.Counter()
    for path in args.inputs:
        for words in read_kaldi_text(path):
            counts.update(words)
    if not counts:
        sys.exit("학습할 어절이 없습니다.")
    model = morfessor.BaselineModel(corpusweight=args.corpus_weight)
    # 많이 나온 어절이 통째로 외워지지 않게 횟수를 로그로 누른다.
    model.load_data([(c, w) for w, c in counts.items()],
                    count_modifier=lambda c: int(round(math.log(c + 1, 2))))
    model.train_batch()
    morfessor.MorfessorIO().write_binary_model_file(args.model, model)
    print(f"어절 {len(counts)}종으로 학습 -> {args.model}", file=sys.stderr)


def segment_word(model, word, cache):
    if word not in cache:
        pieces = model.viterbi_segment(word)[0]
        cache[word] = [pieces[0]] + ["+" + p for p in pieces[1:]]
    return cache[word]


def cmd_apply(args):
    model = morfessor.MorfessorIO().read_binary_model_file(args.model)
    cache = {}
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        head, words = ([parts[0]], parts[1:]) if args.has_id else ([], parts)
        out = head[:]
        for w in words:
            out.extend(segment_word(model, w, cache))
        sys.stdout.write(" ".join(out) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--model", required=True)
    p.add_argument("--corpus-weight", type=float, default=1.0,
                   help="클수록 덜 쪼갠다")
    p.add_argument("inputs", nargs="+",
                   help="Kaldi text 파일(이름이 text로 끝남) 또는 단어 목록")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("apply")
    p.add_argument("--model", required=True)
    p.add_argument("--no-id", dest="has_id", action="store_false",
                   help="줄 앞에 발화 ID가 없는 입력")
    p.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
