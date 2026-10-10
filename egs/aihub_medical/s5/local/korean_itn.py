#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""한글로 읽은 수를 아라비아 숫자로 바꾼다.

AI-Hub 전사는 수를 한글로 적으므로 Kaldi 도 "백사십", "육 점 오" 처럼 낸다.
리포트(-AI 의 voice-analyze)는 검사 수치와 처방 기간을 아라비아 숫자로 찾으므로
그 앞에서 바꿔 준다.

  혈압이 백사십에 구십이에요     -> 혈압이 140에 90이에요
  당화혈색소는 육 점 오예요      -> 당화혈색소는 6.5예요
  삼 개월 치 드릴게요            -> 3개월 치 드릴게요

수와 소리가 같은 낱말이 많다(이 분, 사는, 오늘, 백신, 천식, 사십견).
그래서 확실한 것만 바꾼다.
  - 자리 단위(십 백 천 만 억)가 든 두 글자 이상이면 수로 본다("이십", "백사").
  - 한 글자 수는 단위가 뒤따를 때만 바꾼다("삼 개월", "오 분", "육 점 오").
  - "이"는 "이 분", "이 점"처럼 가리키는 말이 많아 몇몇 단위에서는 그대로 둔다.
  - 수 뒤에 붙은 말은 조사, 어미, 단위일 때만 바꾼다("사십견"은 그대로).
읽는 법이 둘인 곳은 이렇게 정했다.
  - "이십 일"은 20일, "이십일"은 21 로 읽는다(띄어 쓴 '일'은 날짜·기간).
  - "구십이에요"는 90이에요("이에요"가 받침 뒤에 오는 꼴, 92 면 "구십이예요").
  - "백사십이 넘으면", "이백 사십이고"처럼 끝의 '이'가 조사나 '이다'로 읽히면
    딱 떨어지는 수(140, 240)로 본다. 의사가 기준값을 말할 때 흔한 꼴이다.
틀릴 수 있으므로 raw_text 에는 바꾸기 전 글을 남긴다.

  python3 local/korean_itn.py "혈압이 백사십에 구십이에요"
  python3 local/korean_itn.py --test
"""

import re
import sys

DIGITS = {"영": 0, "공": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
          "육": 6, "륙": 6, "칠": 7, "팔": 8, "구": 9}
SMALL = {"십": 10, "백": 100, "천": 1000}
LARGE = {"만": 10 ** 4, "억": 10 ** 8}
MONTHS = {"시월": "10월", "유월": "6월"}

COUNTERS = ["개월", "주일", "주간", "주", "년", "일", "회", "차", "기", "단계", "등급",
            "퍼센트", "프로", "밀리그램", "미리그램", "밀리리터", "밀리", "미리", "그램",
            "키로", "킬로", "센티", "센치", "미터", "리터", "시시", "씨씨", "도", "정",
            "점", "분", "초", "시간", "월", "층", "세", "원", "대"]
# "이"(this) 뒤에 오면 수가 아닐 때가 많은 단위. "이 분", "이 점", "이 일"
NOT_AFTER_I = {"일", "분", "시간", "정", "점", "도", "차", "기", "세", "초", "원", "대"}
# 한 글자 수에 붙여 써도 수로 볼 단위. "삼개월", "일주일", "사월". "오일"(기름),
# "사주", "이분" 같은 낱말과 겹치는 한 글자 단위는 뺐다.
ATTACHED = [c for c in COUNTERS if len(c) >= 2] + ["월", "년"]

PARTICLES = ["이", "가", "은", "는", "을", "를", "에", "에서", "에게", "으로", "로", "와",
             "과", "하고", "이나", "나", "이랑", "랑", "도", "만", "까지", "부터", "보다",
             "정도", "쯤", "씩", "대로", "가량", "짜리", "째", "간", "동안", "치", "분",
             "이상", "이하", "이내", "미만", "초과", "밖에", "뿐", "마다"]
ENDINGS = ["이요", "요", "이에요", "예요", "이고", "고", "이니까", "니까", "이라", "라",
           "이라고", "라고", "이라서", "라서", "이라면", "라면", "이면", "면", "이며", "며",
           "이네요", "네요", "이죠", "죠", "이지", "지", "이지요", "지요", "이었어요",
           "였어요", "이었고", "였고", "이었는데", "였는데", "이었", "였", "이야", "야",
           "입니다", "이구요", "구요", "인데", "인데요", "인가요"]


def _alt(words):
    return "|".join(sorted(map(re.escape, words), key=len, reverse=True))


TAIL = rf"(?:{_alt(PARTICLES)})*(?:{_alt(ENDINGS)})?"
SUFFIX_RE = re.compile(rf"(?:{_alt(COUNTERS)})?{TAIL}")
TAIL_RE = re.compile(TAIL)
COUNTER_TOKEN_RE = re.compile(rf"({_alt(COUNTERS)}){TAIL}")
ATTACHED_RE = re.compile(rf"(?:{_alt(ATTACHED)}){TAIL}")
DIGITS_ONLY_RE = re.compile(rf"([{''.join(DIGITS)}]+)({TAIL})")
# "백사십이 넘으면"의 '이'를 조사로 볼 만한 뒷말
PREDICATE_RE = re.compile(r"넘|나오|나와|나왔|되|돼|됐|높|낮|떨어|올라|정상")
STOPWORDS = {"천만에요", "천만에", "천만다행"}
# 받침 뒤의 '이다'. 끝의 '이'가 이것의 머리로 읽힐 때만 딱 떨어지는 수로 본다.
COPULA_RE = re.compile(_alt([e for e in ENDINGS if e.startswith("이")]))


def value(text):
    """한자어 수를 읽는다. "이백사십" -> 240. 수가 아니면 None"""
    if not text:
        return None
    if text in ("영", "공"):
        return 0
    total, section, cur = 0, 0, None
    last_small, last_large = 10 ** 5, 10 ** 9
    for ch in text:
        if ch in DIGITS:
            if cur is not None or DIGITS[ch] == 0:
                return None
            cur = DIGITS[ch]
        elif ch in SMALL:
            unit = SMALL[ch]
            if unit >= last_small:
                return None
            section += (cur or 1) * unit
            cur, last_small = None, unit
        elif ch in LARGE:
            unit = LARGE[ch]
            if unit >= last_large:
                return None
            section += cur or 0
            total += (section or 1) * unit
            section, cur, last_small, last_large = 0, None, 10 ** 5, unit
        else:
            return None
    return total + section + (cur or 0)


def strong(num):
    """단위가 든 두 글자 이상. 낱말과 겹칠 일이 거의 없다."""
    return len(num) >= 2 and any(ch in SMALL or ch in LARGE for ch in num)


def round_alt(num, suffix, nxt):
    """끝의 '이'를 조사나 '이다'로 읽을 수 있으면 딱 떨어지는 쪽을 돌려준다."""
    head = num[:-1]
    if not (num.endswith("이") and len(head) >= 2 and head[-1] in SMALL.keys() | LARGE.keys()):
        return None
    if suffix and COPULA_RE.fullmatch("이" + suffix):
        return head, "이" + suffix
    if not suffix and nxt and PREDICATE_RE.match(nxt):
        return head, "이"
    return None


def read_number(tokens, i):
    """tokens[i] 부터 수 하나를 읽는다. (수 글자, 뒤에 붙은 말, 쓴 토큰 수) 또는 None

    띄어 쓴 수는 이어 읽는다("이백 십 육" -> 이백십육). 마지막 토큰만 뒷말을 가질 수 있다.
    """
    best = None
    num = ""
    k = i
    while k < len(tokens):
        tok = tokens[k]
        # 띄어 쓴 '일'은 날짜·기간 단위로 본다. "이십 일" -> 20일
        if k > i and tok.startswith("일"):
            break
        for cut in range(len(tok), 0, -1):
            part, suffix = tok[:cut], tok[cut:]
            if value(num + part) is None:
                continue
            if suffix and not SUFFIX_RE.fullmatch(suffix):
                continue
            best = (num + part, suffix, k - i + 1)
            break
        if value(num + tok) is None:
            break
        num += tok
        k += 1
    return best


def convert(text):
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in STOPWORDS:
            out.append(tok)
            i += 1
            continue
        month = next((m for m in MONTHS if tok.startswith(m) and TAIL_RE.fullmatch(tok[len(m):])), None)
        if month:
            out.append(MONTHS[month] + tok[len(month):])
            i += 1
            continue
        found = read_number(tokens, i)
        if not found:
            out.append(tok)
            i += 1
            continue
        num, suffix, used = found
        nxt = tokens[i + used] if i + used < len(tokens) else ""
        after = tokens[i + used + 1] if i + used + 1 < len(tokens) else ""

        alt = round_alt(num, suffix, nxt)
        if alt:
            num, suffix = alt

        # 소수: "육 점 오", "영 점 오"
        frac = DIGITS_ONLY_RE.fullmatch(after) if nxt in ("점", "쩜") and not suffix else None
        if frac and all(DIGITS[c] or c in "영공" for c in frac.group(1)):
            digits = "".join(str(DIGITS[c]) for c in frac.group(1))
            out.append(f"{value(num)}.{digits}{frac.group(2)}")
            i += used + 2
            continue

        counter = COUNTER_TOKEN_RE.fullmatch(nxt) if not suffix else None
        if counter and not strong(num) and num == "이" and counter.group(1) in NOT_AFTER_I:
            counter = None
        if counter:
            out.append(f"{value(num)}{nxt}")
            i += used + 1
            continue

        if strong(num) or (suffix and ATTACHED_RE.fullmatch(suffix) and num + suffix[:1] != "이년"):
            out.append(f"{value(num)}{suffix}")
            i += used
            continue

        out.append(tok)
        i += 1
    return " ".join(out)


CASES = [
    ("혈압이 백사십에 구십이에요", "혈압이 140에 90이에요"),
    ("이백 십 육이니까", "216이니까"),
    ("당화혈색소는 육 점 오예요", "당화혈색소는 6.5예요"),
    ("영 점 오 밀리", "0.5 밀리"),
    ("삼 개월 치 드릴게요", "3개월 치 드릴게요"),
    ("이 주 뒤에 오세요", "2주 뒤에 오세요"),
    ("삼십 분 정도", "30분 정도"),
    ("사 점 정도 아파요", "4점 정도 아파요"),
    ("이십대 여성", "20대 여성"),
    ("시월 이십 일에 오세요", "10월 20일에 오세요"),
    ("시월 이십일 일에", "10월 21일에"),
    ("일주일에 한 번", "1주일에 한 번"),
    ("백사십이 넘으면", "140이 넘으면"),
    ("콜레스테롤이 이백 사십이고", "콜레스테롤이 240이고"),
    ("구십이예요", "92예요"),
    ("공복 혈당이 백 이십", "공복 혈당이 120"),
    ("만 이천 보", "12000 보"),
    ("유월에 뵐게요", "6월에 뵐게요"),
    # 그대로 둬야 하는 것
    ("이 분은 괜찮아요", "이 분은 괜찮아요"),
    ("백신 맞으셨죠", "백신 맞으셨죠"),
    ("천천히 드세요", "천천히 드세요"),
    ("사십견이 있어요", "사십견이 있어요"),
    ("오늘 사는 게 힘들어요", "오늘 사는 게 힘들어요"),
    ("이 점 이해해 주세요", "이 점 이해해 주세요"),
    ("천만에요", "천만에요"),
    ("팔이 아파요", "팔이 아파요"),
    ("이번에 오일 드세요", "이번에 오일 드세요"),
    ("만성 위염", "만성 위염"),
    ("이상 없어요", "이상 없어요"),
]


def self_test():
    bad = 0
    for given, want in CASES:
        got = convert(given)
        if got != want:
            bad += 1
            print(f"틀림: {given!r} -> {got!r} (기대 {want!r})")
    print(f"{len(CASES) - bad}/{len(CASES)} 통과")
    return bad == 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        sys.exit(0 if self_test() else 1)
    lines = sys.argv[1:] or sys.stdin
    for line in lines:
        print(convert(line.strip()))
