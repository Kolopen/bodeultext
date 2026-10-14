#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""Kaldi 데이터 폴더(data/train, data/test)를 whisper 학습용 목록(TSV)으로 바꾼다.

  python3 local/whisper/prepare_data.py data/train exp/whisper/train.tsv \\
      --labels /mnt/e/aihub/train/labels --max-utts 100000 --pack-sec 28 --holdout data/test/text
  python3 local/whisper/prepare_data.py data/test exp/whisper/all_test.tsv \\
      --labels /mnt/e/aihub/valid/labels

음성과 발화는 Kaldi 와 같은 것을 쓰고, 글은 Kaldi 용(text.word)이 아니라 AI-Hub 라벨의
원래 글에서 만든다. whisper 가 원래 내놓는 꼴에 맞춰야 리포트가 제대로 나온다.
  - 문장부호(. , ? !)를 남긴다. 리포트가 문장을 나눌 때 쓴다.
  - 한글로 적힌 수는 숫자로 바꾼다(local/korean_itn.py, Kaldi 결과에 쓰는 것과 같은 규칙).
    수를 한글로 배우면 "이백 십 육" 처럼 내놓아 리포트가 검사 수치를 못 찾는다
    (-AI README 의 ghost613 비교가 그 경우다).
--labels 를 안 주면 text.word 에서 만든다(문장부호 없음).

--pack-sec 를 주면 같은 화자의 이어진 발화를 그 길이 안에서 이어 붙여 한 줄로 만든다.
whisper 는 늘 30초 단위로 들어서, 3초짜리 발화 하나만 주면 나머지 27초는 빈 소리다.
이어 붙이면 같은 시간에 몇 배를 배우고, 실제 녹음처럼 여러 문장이 든 구간을 끝까지
받아 적도록 배운다(짧은 문장만 배우면 긴 구간에서 첫 문장만 쓰고 멈추기 쉽다).
이을 길이는 줄마다 0 ~ pack-sec 초에서 고르게 뽑으므로 한 문장짜리 줄도 남는다.

--max-per-sentence 는 같은 문장을 몇 번까지 쓸지다. 이 데이터는 한 문장을 수십 명이 읽어서
(의사 세트는 평균 58명) 다 쓰면 모델이 그 문장들을 외워, 실제 진료에서 소리가 흐릴 때 외운
문장으로 채워 넣기 쉽다. 문장마다 여러 목소리를 고르게 뽑고 나머지는 버린다.
--groups 는 화자 구분(발화 ID 앞 두 글자)을 고른다. HB 는 의사, HA 는 간호사, PA~PF 는 환자.

--holdout 으로 평가 문장(Kaldi text)을 주면 같은 문장(군말·띄어쓰기만 다른 것까지)을 다른
사람이 읽은 발화도 학습에서 뺀다. 이 데이터는 한 문장을 여러 사람이 읽어서, 빼지 않으면
평가 문장을 외워서 맞히게 된다. data/train 은 Kaldi 때 이미 평가 문장을 뺐지만, 세트를 더
받아 만든 data/train_all 에는 평가 문장이 다른 사람 목소리로 들어 있다.

한 줄: 발화ID<TAB>wav|wav..<TAB>길이|길이..<TAB>글|글..  (이어 붙인 것은 '|' 로 잇는다)
"""

import argparse
import collections
import random
import re
import shlex
import sys
from itertools import groupby
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from aihub import (DEID_RE, DUAL_RE, FILLER_SLASH_RE, field, iter_labels, normalize,  # noqa: E402
                   sentence_key, utt_id_of)
from korean_itn import convert  # noqa: E402

SEP = "|"
# 문장부호. 숫자 사이의 점·쉼표(38.5, 3,000)는 나누지 않는다.
PUNCT_RE = re.compile(r"((?<!\d)[.,?!]+|[.,?!]+(?!\d))")
# 소리가 없는 표시(용어 표시 * 등)
DROP_RE = re.compile(r"[*~…·\"'`“”‘’:;|]")


def whisper_text(raw):
    """AI-Hub 전사를 whisper 정답으로 바꾼다. 쓸 수 없으면 None."""
    if DEID_RE.search(raw):  # 비식별화 태그: 음성에는 실제 이름이 들어 있다
        return None
    text = DUAL_RE.sub(lambda m: m.group(1), raw)  # 이중전사 (216)/(이백십육) -> 216
    text = FILLER_SLASH_RE.sub("", text)
    text = DROP_RE.sub(" ", text)
    out = []
    for k, part in enumerate(PUNCT_RE.split(text)):
        if k % 2:
            out.append(part)
            continue
        # 수 바꾸기는 문장부호 사이마다 한다. "구십이에요." 의 '.' 이 끝말을 가리지 않게.
        words = convert(" ".join(part.split()))
        if words:
            out.append((" " if out else "") + words)
    return "".join(out).strip() or None


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


def label_texts(root, wanted):
    """라벨 폴더(또는 zip)에서 wanted 발화의 원래 전사를 꺼낸다."""
    texts = {}
    for path, label in iter_labels(root):
        utt = utt_id_of(label, path)
        if utt in wanted:
            texts[utt] = str(field(label, "전사정보", "LabelText"))
    return texts


def pack(rows, pack_sec, rng):
    """같은 화자의 이어진 발화를 0 ~ pack_sec 초에서 뽑은 길이 안으로 묶는다."""
    groups = []
    for _, items in groupby(rows, key=lambda r: r[1]):
        items = list(items)
        i = 0
        while i < len(items):
            target = rng.uniform(0, pack_sec)
            group, total = [items[i]], items[i][3]
            i += 1
            while i < len(items) and items[i][3] > 0 and total + items[i][3] <= target:
                group.append(items[i])
                total += items[i][3]
                i += 1
            groups.append(group)
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, help="Kaldi 데이터 폴더 (wav.scp, text.word, utt2dur, utt2spk)")
    p.add_argument("out", type=Path, help="만들 TSV")
    p.add_argument("--labels", help="AI-Hub 라벨 폴더나 zip (원래 전사를 쓴다. 문장부호를 살리려면 준다)")
    p.add_argument("--max-utts", type=int, default=0, help="이만큼만 고르게 뽑는다 (0 이면 전부)")
    p.add_argument("--max-sec", type=float, default=30.0, help="whisper 는 30초까지만 듣는다")
    p.add_argument("--pack-sec", type=float, default=0.0, help="이어 붙일 최대 길이(초). 0 이면 잇지 않는다")
    p.add_argument("--holdout", action="append", default=[],
                   help="이 Kaldi text(평가 문장)와 같은 문장은 학습에서 뺀다. 여러 번 줄 수 있다")
    p.add_argument("--max-per-sentence", type=int, default=0,
                   help="같은 문장은 이만큼까지만 쓴다(다른 목소리로 고르게). 0 이면 전부")
    p.add_argument("--groups", help="이 화자 구분만 쓴다. 예: HB,PA (HB 의사, HA 간호사, PA~PF 환자)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    held = set()
    for path in args.holdout:
        with open(path, encoding="utf-8") as f:
            held.update(sentence_key(line.split(maxsplit=1)[1]) for line in f if " " in line.strip())

    wavs = read_map(args.data / "wav.scp")
    durs = {k: float(v) for k, v in read_map(args.data / "utt2dur").items()} \
        if (args.data / "utt2dur").is_file() else {}
    spks = read_map(args.data / "utt2spk") if (args.data / "utt2spk").is_file() else {}
    if args.labels:
        if not Path(args.labels).exists():
            sys.exit(f"라벨이 없습니다: {args.labels}\n"
                     "  AI-Hub 를 푼 폴더(--data-root)를 지웠다면 다시 풀어야 합니다(음성도 같이 필요).")
        raw = label_texts(args.labels, set(wavs))
        missing = len(wavs) - len(raw)
        if missing:
            print(f"  라벨을 못 찾은 발화 {missing}개는 뺍니다", file=sys.stderr)
    else:
        text_file = args.data / "text.word"
        if not text_file.is_file():
            text_file = args.data / "text"
        raw = read_map(text_file)
        print("  --labels 가 없어 Kaldi 글(text.word)을 씁니다. 문장부호가 없습니다.", file=sys.stderr)

    groups = set(args.groups.split(",")) if args.groups else None
    rows, changed, dropped, held_out, other = [], [], 0, 0, 0
    for utt in sorted(wavs):
        if utt not in raw:
            continue
        if groups and utt[:2] not in groups:
            other += 1
            continue
        dur = durs.get(utt, 0.0)
        text = whisper_text(raw[utt])
        if text is None or dur > args.max_sec:
            dropped += 1
            continue
        # 같은 문장인지는 Kaldi 와 같은 꼴(수는 한글, 문장부호 없음)로 견준다
        key = sentence_key(normalize(raw[utt])[0] or raw[utt])
        if key in held:
            held_out += 1
            continue
        path = wav_path(wavs[utt])
        if SEP in path:
            sys.exit(f"경로에 '{SEP}' 가 있으면 쓸 수 없습니다: {path}")
        if re.search(r"\d", text) and not re.search(r"\d", raw[utt]):
            changed.append((raw[utt], text))
        rows.append((utt, spks.get(utt, utt.split("-", 1)[0]), path, dur, text, key))
    if not rows:
        sys.exit(f"쓸 발화가 없습니다: {args.data}" + (f" (--groups {args.groups})" if groups else ""))

    rng = random.Random(args.seed)
    same = collections.defaultdict(list)
    for r in rows:
        same[r[5]].append(r)
    capped = 0
    if args.max_per_sentence:
        rows = []
        for key in sorted(same):
            readings = same[key]
            if len(readings) > args.max_per_sentence:
                capped += len(readings) - args.max_per_sentence
                readings = rng.sample(readings, args.max_per_sentence)
            rows.extend(readings)
        rows.sort()
    if args.max_utts and len(rows) > args.max_utts:
        rows = sorted(rng.sample(rows, args.max_utts))
    rows.sort(key=lambda r: (r[1], r[0]))  # 화자별로, 그 안에서는 발화 순서대로
    lines = pack(rows, args.pack_sec, rng) if args.pack_sec > 0 else [[r] for r in rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for g in lines:
            name = g[0][0] + (f"+{len(g) - 1}" if len(g) > 1 else "")
            f.write("\t".join([name, SEP.join(r[2] for r in g), SEP.join(f"{r[3]:.2f}" for r in g),
                               SEP.join(r[4] for r in g)]) + "\n")

    hours = sum(r[3] for r in rows) / 3600
    print(f"{args.out}: 발화 {len(rows)}개, {hours:.1f}시간, 서로 다른 문장 {len({r[5] for r in rows})}개"
          f" -> {len(lines)}줄"
          + (f" (한 줄에 평균 {len(rows) / len(lines):.1f}개를 이어 붙임)" if args.pack_sec > 0 else ""))
    if capped:
        print(f"  같은 문장은 {args.max_per_sentence}번까지만 써서 뺀 발화 {capped}개")
    if other:
        print(f"  다른 화자 구분이라 뺀 발화 {other}개 (--groups {args.groups})")
    if dropped:
        print(f"  뺀 발화 {dropped}개 (비식별화 태그, {args.max_sec:.0f}초 넘음)")
    if held_out:
        print(f"  평가 문장과 같은 문장이라 뺀 발화 {held_out}개")
    if changed:
        print(f"  수를 숫자로 바꾼 발화 {len(changed)}개. 예:")
        for before, after in random.Random(1).sample(changed, min(8, len(changed))):
            print(f"    {before}\n      -> {after}")


if __name__ == "__main__":
    main()
