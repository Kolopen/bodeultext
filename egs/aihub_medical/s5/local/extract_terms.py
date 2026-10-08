#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""-AI 저장소의 진료과 용어 사전(data/terms/*.txt)에서 단어만 뽑는다.

사전 파일은 [drug], [condition] 같은 구역 아래 한 줄에 한 용어를 적고,
[misheard] 구역에는 "침해 = 치매"처럼 오인식 짝을 적는다. 여기서는 바른
용어만 필요하므로 "=" 왼쪽(틀린 말)은 버리고 오른쪽만 쓴다.
한글로만 된 용어만 남긴다. 영문 약어(MRI)는 아직 발음사전에 넣지 못한다.

  python3 local/extract_terms.py ~/-AI/src/voice_ai/data/terms > data/local/terms.txt
"""

import os
import re
import sys

HANGUL = re.compile(r"^[가-힣]+$")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    terms = set()
    for name in sorted(os.listdir(root)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(root, name), encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line or line.startswith("["):
                    continue
                if "=" in line:
                    line = line.split("=", 1)[1].strip()
                for word in line.split():
                    if HANGUL.match(word):
                        terms.add(word)
    for term in sorted(terms):
        print(term)
    print(f"용어 {len(terms)}개", file=sys.stderr)


if __name__ == "__main__":
    main()
