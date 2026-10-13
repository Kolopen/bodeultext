#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""받아 적기 엔진들을 정답(클로바노트 등)과 견준다.

  python3 local/compare_asr.py /mnt/e/녹음/0102.m4a /mnt/e/녹음/0103.m4a ...

녹음마다 이런 파일을 찾는다 (json 은 local/eval_engines.sh 가 만든다).
  /mnt/e/녹음/0102_대본.txt         실제로 한 말(원본 대본). 있으면 이것이 정답이다
  /mnt/e/녹음/0102.txt              클로바노트에서 내보낸 텍스트. 대본이 없으면 이것이 정답,
                                    대본이 있으면 클로바노트도 한 엔진으로 함께 잰다
  /mnt/e/녹음/0102/kaldi.json       Kaldi
  /mnt/e/녹음/0102/whisper.json     whisper turbo
  /mnt/e/녹음/0102/sense.json       SenseVoice (있으면)

재는 것
  글자 오류율   띄어쓰기·문장부호를 빼고 한글 글자만 견준다. 숫자는 한글로 읽어서
                견준다("216" 과 "이백십육"을 같게 본다).
  숫자 맞음     정답에 나온 숫자 중 받아 적은 글에도 나온 수
  없는 숫자     받아 적은 글에만 있는 숫자. 리포트에 가짜 수치로 오를 수 있다.
글자 오류율은 엔진끼리 견주는 용도다. 정답도 사람이 다듬은 글이라 완벽하지 않다.
"""

import argparse
import collections
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 대본 줄 앞의 화자·시각 표시. "의사: ", "환자 : ", "A:", "[00:12] ", "00:12 "
LABEL_RE = re.compile(r"^\s*(?:\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*)?(?:[^\s:：]{1,8}\s*\d?\s*[:：]\s*)?")
CLOVA_HEADER = re.compile(r"^참석자\s*\d+\s+(?:\d{1,2}:)?\d{1,2}:\d{2}\s*$", re.M)
ENGINES = ("kaldi", "whisper", "sense")
DIGITS = "영일이삼사오육칠팔구"


def import_load():
    try:
        from voice_ai.analyze import load
    except ImportError:
        root = Path.home() / "-AI" / "src"
        sys.path.insert(0, str(root))
        try:
            from voice_ai.analyze import load
        except ImportError as e:
            sys.exit(f"-AI 의 voice_ai 를 불러올 수 없습니다 ({e}). 다음단계.txt 의 [6-1] 참고")
    return load


def read_int(n):
    """정수를 한자어로 읽는다. 216 -> 이백십육"""
    if n == 0:
        return "영"

    def four(x):
        out = ""
        for unit, ch in ((1000, "천"), (100, "백"), (10, "십")):
            d = x // unit % 10
            if d:
                out += ("" if d == 1 else DIGITS[d]) + ch
        return out + (DIGITS[x % 10] if x % 10 else "")

    out = ""
    for unit, ch in ((10 ** 8, "억"), (10 ** 4, "만")):
        part = n // unit % 10 ** 4
        if part:
            out += ("" if part == 1 and ch == "만" else four(part)) + ch
    return out + four(n % 10 ** 4)


def spell_numbers(text):
    def swap(m):
        whole, _, frac = m.group().partition(".")
        out = read_int(int(whole))
        if frac:
            out += "점" + "".join(DIGITS[int(c)] for c in frac)
        return out
    return re.sub(r"\d+(?:\.\d+)?", swap, text.replace(",", ""))


def hangul(text):
    return [c for c in spell_numbers(text) if "가" <= c <= "힣"]


def numbers(text):
    """숫자 목록. 대본처럼 숫자를 한글로 적은 글("이백십육")도 숫자로 바꿔서 센다."""
    sys.path.insert(0, str(HERE))
    from korean_itn import convert
    return re.findall(r"\d+(?:\.\d+)?", convert(text.replace(",", "")))


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def full_text(path, load):
    if path.suffix == ".txt":
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        if not CLOVA_HEADER.search(raw):  # 화자·시각 표시만 떼고 그대로 쓴다
            return " ".join(LABEL_RE.sub("", line) for line in raw.splitlines())
    utterances, _ = load(path)
    return " ".join(u.text for u in sorted(utterances, key=lambda u: u.start_ms))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", nargs="+", type=Path, help="녹음 파일들 (옆에 같은 이름의 .txt 정답이 있어야 한다)")
    args = p.parse_args()
    load = import_load()

    rows, total = [], collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for audio in args.audio:
        script = audio.with_name(audio.stem + "_대본.txt")
        clova = audio.with_suffix(".txt")
        ref_path = script if script.is_file() else clova
        if not ref_path.is_file():
            print(f"정답이 없어 건너뜁니다: {script.name} / {clova.name}", file=sys.stderr)
            continue
        print(f"{audio.stem}: 정답 = {ref_path.name}", file=sys.stderr)
        ref = full_text(ref_path, load)
        ref_chars, ref_nums = hangul(ref), numbers(ref)
        hyps = [(e, audio.with_suffix("") / f"{e}.json") for e in ENGINES]
        if ref_path == script:
            hyps.insert(0, ("clova", clova))
        for engine, hyp_path in hyps:
            if not hyp_path.is_file():
                continue
            hyp = full_text(hyp_path, load)
            errors = edit_distance(ref_chars, hangul(hyp))
            want, got = collections.Counter(ref_nums), collections.Counter(numbers(hyp))
            hit = sum((want & got).values())
            extra = sum(got.values()) - hit
            rows.append((audio.stem, engine, errors, len(ref_chars), hit, len(ref_nums), extra))
            t = total[engine]
            for k, v in enumerate((errors, len(ref_chars), hit, len(ref_nums), extra)):
                t[k] += v
    if not rows:
        sys.exit("견줄 것이 없습니다. local/eval_engines.sh 를 먼저 돌리세요.")

    print(f"{'녹음':<12}{'엔진':<9}{'글자 오류율':>10}{'숫자 맞음':>12}{'없는 숫자':>10}")
    for name, engine, errors, n, hit, want, extra in rows:
        print(f"{name:<12}{engine:<9}{100 * errors / max(n, 1):>9.1f}%{f'{hit}/{want}':>12}{extra:>10}")
    if len({r[0] for r in rows}) > 1:
        print("-" * 53)
        for engine, (errors, n, hit, want, extra) in total.items():
            print(f"{'합계':<12}{engine:<9}{100 * errors / max(n, 1):>9.1f}%{f'{hit}/{want}':>12}{extra:>10}")
    print("\n글자 오류율·없는 숫자는 낮을수록, 숫자 맞음은 높을수록 좋다.")


if __name__ == "__main__":
    main()
