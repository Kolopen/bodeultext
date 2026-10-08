#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""한글 표기에서 발음사전을 만든다.

영어처럼 사람이 만든 발음사전이 없으므로 글자에서 바로 발음을 만든다.
한글은 소리와 글자가 거의 맞아서 이 방식이 잘 먹힌다. 다만 그대로 쓰면
어긋나는 자리가 둘 있어 그것만 규칙으로 맞춘다.

  받침은 일곱 소리로 난다    부엌 -> 부억, 옷 -> 옫, 값 -> 갑
  받침은 모음 앞에서 넘어간다  얼굴이 -> 얼구리, 많이 -> 마니, 좋아 -> 조아

넘어가기는 한 단위(조각) 안에서만 한다. "얼굴 +이"처럼 갈라진 경계는
트라이폰 모델이 앞뒤 소리를 보고 배운다.

  python3 local/prepare_dict.py data/local/dict < 단위목록
"""

import os
import sys

INITIALS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "",
            "j", "jj", "ch", "k", "t", "p", "h"]
VOWELS = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae",
          "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i"]
CODAS = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ",
         "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
         "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

# 받침의 대표음. 초성과 이름을 달리해 따로 배우게 한다.
CODA_SOUND = {
    "ㄱ": "K", "ㄲ": "K", "ㄳ": "K", "ㄺ": "K", "ㅋ": "K",
    "ㄴ": "N", "ㄵ": "N", "ㄶ": "N",
    "ㄷ": "T", "ㅅ": "T", "ㅆ": "T", "ㅈ": "T", "ㅊ": "T", "ㅌ": "T", "ㅎ": "T",
    "ㄹ": "L", "ㄼ": "L", "ㄽ": "L", "ㄾ": "L", "ㅀ": "L",
    "ㅁ": "M", "ㄻ": "M",
    "ㅂ": "P", "ㅄ": "P", "ㄿ": "P", "ㅍ": "P",
    "ㅇ": "NG",
}

# 모음 앞에서 받침이 넘어갈 때: (남는 받침, 넘어가는 초성)
LIAISON = {
    "ㄱ": ("", "g"), "ㄲ": ("", "kk"), "ㄴ": ("", "n"), "ㄷ": ("", "d"),
    "ㄹ": ("", "r"), "ㅁ": ("", "m"), "ㅂ": ("", "b"), "ㅅ": ("", "s"),
    "ㅆ": ("", "ss"), "ㅈ": ("", "j"), "ㅊ": ("", "ch"), "ㅋ": ("", "k"),
    "ㅌ": ("", "t"), "ㅍ": ("", "p"),
    "ㅎ": ("", ""),              # 좋아 -> 조아
    "ㄶ": ("", "n"),             # 많이 -> 마니
    "ㅀ": ("", "r"),             # 싫어 -> 시러
    "ㄳ": ("ㄱ", "s"), "ㄵ": ("ㄴ", "j"), "ㄺ": ("ㄹ", "g"), "ㄻ": ("ㄹ", "m"),
    "ㄼ": ("ㄹ", "b"), "ㄽ": ("ㄹ", "s"), "ㄾ": ("ㄹ", "t"), "ㄿ": ("ㄹ", "p"),
    "ㅄ": ("ㅂ", "s"),
}


def decompose(syllable):
    code = ord(syllable) - 0xAC00
    if not 0 <= code < 11172:
        raise ValueError(f"한글 음절이 아닙니다: {syllable!r}")
    return INITIALS[code // 588], VOWELS[(code % 588) // 28], CODAS[code % 28]


def pronounce(unit):
    word = unit.lstrip("+")
    sylls = [list(decompose(ch)) for ch in word]
    for i in range(len(sylls) - 1):
        coda = sylls[i][2]
        if coda and coda != "ㅇ" and sylls[i + 1][0] == "" and coda in LIAISON:
            sylls[i][2], sylls[i + 1][0] = LIAISON[coda]
    phones = []
    for initial, vowel, coda in sylls:
        if initial:
            phones.append(initial)
        phones.append(vowel)
        if coda:
            phones.append(CODA_SOUND[coda])
    return phones


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)

    units = sorted({w for line in sys.stdin for w in line.split()})
    lexicon = []
    used = set()
    skipped = 0
    for unit in units:
        try:
            phones = pronounce(unit)
        except ValueError:
            skipped += 1
            continue
        if not phones:
            skipped += 1
            continue
        used.update(phones)
        lexicon.append(f"{unit} {' '.join(phones)}")

    def write(name, lines):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    write("lexicon.txt", ["!SIL SIL", "<UNK> SPN"] + lexicon)
    write("silence_phones.txt", ["SIL", "SPN"])
    write("optional_silence.txt", ["SIL"])
    write("nonsilence_phones.txt", sorted(used))
    write("extra_questions.txt", [])
    print(f"{out}: 단위 {len(lexicon)}개, 음소 {len(used)}개"
          + (f", 한글이 아니라 뺀 것 {skipped}개" if skipped else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
