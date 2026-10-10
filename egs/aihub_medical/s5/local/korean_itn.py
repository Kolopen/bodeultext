#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""한글로 읽은 수를 아라비아 숫자로 바꾼다.

AI-Hub 전사는 수를 한글로 적으므로 Kaldi 도 "백사십", "육 점 오" 처럼 낸다.
리포트(-AI 의 voice-analyze)는 검사 수치와 처방 기간을 아라비아 숫자로 찾으므로
그 앞에서 바꿔 준다.

  혈압이 백사십에 구십이에요     -> 혈압이 140에 90이에요
  당화혈색소는 육 점 오고요      -> 당화혈색소는 6.5 고요
  삼 개월 치 드릴게요            -> 3개월 치 드릴게요

수와 소리가 같은 낱말이 많다(이 분, 사는, 오늘, 백신, 천식, 사십견, 만일, 이천).
틀린 숫자가 리포트에 오르는 것이 숫자를 놓치는 것보다 나쁘므로, 헷갈리면 그대로 둔다.
  - 자리 단위(십 백 천 만 억)가 든 두 글자 이상이면 수로 본다("이십", "백사").
    다만 땅 이름이나 낱말(이천, 천사, 백일, 만일)은 단위가 붙을 때만 바꾼다.
  - 한 글자 수는 단위가 뒤따를 때만 바꾼다("삼 개월", "오 분", "육 점 오").
    "이"(이 분, 이 단계)와 "일"(무슨 일, 일 정도)은 몇몇 단위 앞에서만 바꾼다.
  - 수 뒤에 붙은 말은 조사, 어미, 단위일 때만 바꾼다("사십견"은 그대로).
  - "오 육 개월", "일 이 년" 같은 어림은 바꾸지 않는다.
읽는 법이 둘인 곳은 이렇게 정했다.
  - "이십 일"은 20일. 붙여 쓴 "이십일"은 뒤에 치·분·동안·뒤·후·날이 오거나 앞에
    달(시월, 다음 달)이 있으면 20일, "이십일에"처럼 날짜로도 수로도 읽히면 그대로,
    그 밖에는 21.
  - "구십이에요"는 90이에요("이에요"가 받침 뒤에 오는 꼴, 92 면 "구십이예요").
  - "백사십이 넘으면", "이백 사십이고", "팔십이세요"처럼 끝의 '이'가 조사나
    '이다'로 읽히면 딱 떨어지는 수(140, 240, 80)로 본다. 기준값을 말할 때 흔한 꼴이다.
  - "만 육십오 세"의 '만'은 만 나이라 그대로 두고 뒤의 수만 바꾼다.
-AI 는 숫자 바로 뒤에 붙은 낯선 한글을 깨진 낱말로 보고 숫자를 지우므로, 그런
어미는 한 칸 띄어 둔다("92 예요", "6.5 고요"). 단위는 붙여 쓴다("5밀리그램").
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
UNITS = set(SMALL) | set(LARGE)
MONTHS = {"시월": "10월", "유월": "6월"}

COUNTERS = ["개월", "주일", "주간", "주", "년", "일", "회", "차", "기", "단계", "등급",
            "퍼센트", "프로", "밀리그램", "미리그램", "밀리리터", "밀리미터", "미리미터",
            "센티미터", "밀리", "미리", "그램",
            "키로", "킬로", "센티", "센치", "미터", "리터", "시시", "씨씨", "도", "정",
            "점", "분", "초", "시간", "월", "층", "세", "원", "대"]
# "이"(this) 뒤에 오면 수가 아닐 때가 많은 단위. "이 분", "이 점", "이 단계"
NOT_AFTER_I = {"일", "분", "시간", "정", "점", "도", "차", "기", "세", "초", "원", "대",
               "단계", "등급", "층", "프로"}
# "일"(일, work)은 흔한 낱말이라 이 단위 앞에서만 1 로 본다. "무슨 일 주로", "일 시간"
AFTER_IL = {"년", "개월", "주일", "회", "차", "기", "단계", "등급", "퍼센트", "프로",
            "밀리그램", "미리그램", "밀리리터", "밀리미터", "미리미터", "센티미터", "밀리",
            "그램", "키로", "킬로", "센티",
            "센치", "미터", "리터", "시시", "씨씨", "월", "층"}
# 단위로 시작하지만 단위가 아닌 낱말. 정도(정+도), 주로(주+로), 도와(도+와), 세요
NOT_COUNTER_RE = re.compile(r"정도|주로|도와|도움|세요")
# 한 글자 수에 붙여 써도 수로 볼 단위. "삼개월", "일주일", "사월". "오일"(기름),
# "사주", "이분" 같은 낱말과 겹치는 한 글자 단위는 뺐다.
ATTACHED = [c for c in COUNTERS if len(c) >= 2] + ["월", "년"]
# 단위 없이 쓰면 수가 아닐 때가 많은 낱말 (땅 이름, 기념일, '만일')
LEXICAL = {"이천", "사천", "오천", "천사", "백구", "백일", "천일", "만일"}
STOPWORDS = {"천만에요", "천만에", "천만다행"}

PARTICLES = ["이", "가", "은", "는", "을", "를", "에", "에서", "에게", "으로", "로", "와",
             "과", "하고", "이나", "나", "이랑", "랑", "도", "만", "까지", "부터", "보다",
             "정도", "쯤", "씩", "대로", "가량", "짜리", "째", "간", "동안", "치", "분",
             "이상", "이하", "이내", "미만", "초과", "밖에", "뿐", "마다"]
# 어미는 '이다'의 줄기(이, 이었, 이시 ...)에 끝말을 붙여 만든다. 받침 뒤에는 '이'가
# 붙고("구십이고"), 모음 뒤에는 빠진다("구십이고" 의 '구십이' + '고").
_FINAL = ["요", "고", "고요", "니까", "니까요", "라", "라고", "라고요", "라서", "라서요",
          "라면", "면", "면요", "며", "네", "네요", "죠", "지", "지요", "야", "거든요",
          "던데", "던데요", "구요", "든지"]
_PAST = ["어요", "어", "고", "고요", "는데", "는데요", "거든요", "습니다", "죠", "지요",
         "네요", "다"]
_HONOR = ["고", "고요", "죠", "네요", "면", "니까", "는데", "는데요", "지요"]
ENDINGS = (["예요", "이에요", "입니다", "입니까", "인데", "인데요", "인가요", "인지", "이다",
            "이세요", "세요", "이셨어요", "셨어요"]
           + [s + f for s in ("이", "") for f in _FINAL]
           + [s + f for s in ("이었", "였") for f in _PAST]
           + [s + f for s in ("이시", "시") for f in _HONOR])


def _alt(words):
    return "|".join(sorted(map(re.escape, set(words)), key=len, reverse=True))


TAIL = rf"(?:{_alt(PARTICLES)})*(?:{_alt(ENDINGS)})?"
SUFFIX_RE = re.compile(rf"(?:{_alt(COUNTERS)})?{TAIL}")
TAIL_RE = re.compile(TAIL)
COUNTER_TOKEN_RE = re.compile(rf"({_alt(COUNTERS)}){TAIL}")
COUNTER_START_RE = re.compile(_alt(COUNTERS))
ATTACHED_RE = re.compile(rf"(?:{_alt(ATTACHED)}){TAIL}")
DIGITS_ONLY_RE = re.compile(rf"([{''.join(DIGITS)}]+)({TAIL})")
# 받침 뒤의 '이다'와 '이랑', '이나'. 끝의 '이'가 이것의 머리로 읽힐 때 딱 떨어지는 수로 본다.
COPULA_RE = re.compile(_alt([e for e in ENDINGS if e.startswith("이")] + ["이랑", "이나"]))
# "백사십이 넘으면"의 '이'를 조사로 볼 만한 뒷말
PREDICATE_RE = re.compile(r"넘|나오|나와|나왔|되|돼|됐|높|낮|떨어|올라|정상")
# "삼십일 치", "이십일 뒤"의 '일'은 날짜·기간이다
DAY_MARK_RE = re.compile(r"치|분|간|동안|째|뒤|후|만에|날")
# 날짜로도 수로도 읽히는 뒷말. "이십일에" 는 20일에도 21에도 된다
DAY_OR_NUMBER_RE = re.compile(r"에|까지|부터|쯤|에는|에도")
MONTH_RE = re.compile(r"(?:\d+월|달)$")
# 소수점 뒤에 와도 되는 '이'로 시작하는 말. "이십 점 이상이면"
AFTER_POINT_OK_RE = re.compile(r"이(?:상|하|내|후|전|면|고|요|에요|었|야|라|니)")
# -AI(voice_ai/schedule.py 의 _NUMBER_TAIL)는 숫자 바로 뒤에 이 글자들 말고 다른
# 한글이 붙어 있으면 깨진 낱말로 보고 그 숫자를 지운다. 그런 말은 한 칸 띄운다.
AI_TAIL = set("이가은는을를에의도과와로랑만부까나든"
              "점번개명초분시년월일주달회알정씩대배차"
              "였됐됩입쯤여")


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
    return len(num) >= 2 and any(ch in UNITS for ch in num)


def attach(number, rest):
    """숫자 뒤에 남은 말을 붙인다.

    단위는 붙여 쓴다("5밀리그램"). 서술어 꼴의 어미가 -AI 가 지우는 글자로 시작하면
    한 칸 띄운다("92 예요"). -AI 는 단위가 붙은 숫자(약 용량 등)를 검사 수치로 보지
    않는데, 띄우면 그 구분이 사라져 용량이 검사 수치로 잘못 오른다.
    """
    if rest.startswith(("퍼센트", "프로")):  # 당화혈색소, 산소포화도. 용량에는 안 쓴다
        return f"{number} {rest}"
    if (rest and "가" <= rest[0] <= "힣" and rest[0] not in AI_TAIL
            and not COUNTER_START_RE.match(rest)):
        return f"{number} {rest}"
    return f"{number}{rest}"


def week(counter, following):
    """'N주일 뒤'는 'N주 뒤'로 쓴다. -AI 가 'N주일 뒤'를 처방 기간 N주로 잘못 읽는다."""
    if counter.startswith("주일") and re.match(r"뒤|후", following):
        return "주" + counter[2:]
    return counter


def round_alt(num, suffix, nxt):
    """끝의 '이'를 조사나 '이다'로 읽을 수 있으면 딱 떨어지는 쪽을 돌려준다."""
    head = num[:-1]
    if not (num.endswith("이") and (len(head) >= 2 or head in ("백", "천", "만"))
            and head[-1] in UNITS):
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


def counter_of(num, tok):
    """tok 이 num 뒤에 띄어 쓴 단위면 그 단위를 돌려준다."""
    if not tok or NOT_COUNTER_RE.match(tok):
        return None
    m = COUNTER_TOKEN_RE.fullmatch(tok)
    if not m:
        return None
    c = m.group(1)
    if not strong(num):
        if c == "대":  # "삼 대 일"(비율). "이십 대"는 된다
            return None
        if num == "이" and c in NOT_AFTER_I:
            return None
        if num == "일" and c not in AFTER_IL:
            return None
    return c


def read_fraction(tokens, k):
    """소수점 뒤의 숫자들. "점 이 오 밀리" -> ("25", "", 다음 위치)"""
    digits, tail = "", ""
    while k < len(tokens):
        m = DIGITS_ONLY_RE.fullmatch(tokens[k])
        if not m:
            break
        digits += "".join(str(DIGITS[c]) for c in m.group(1))
        tail = m.group(2)
        k += 1
        if tail:
            break
    return digits, tail, k


def keep_man(tokens, i):
    """'만 육십오 세', '만 이 년'의 '만'(만 나이)이면 True. '만 이천'은 수다."""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    if nxt[:1] in ("점", "쩜", "일"):  # "만 점에"(만점), "만 일 년"
        return True
    found = read_number(tokens, i + 1)
    if not found or value(found[0]) >= 10000:
        return False
    num, suffix, used = found
    after = tokens[i + 1 + used] if i + 1 + used < len(tokens) else ""
    return bool(re.match(r"세|살|개월|년|주|일", suffix or after))


def convert(text):
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        prev = out[-1] if out else ""

        def keep():
            out.append(tok)

        if tok in STOPWORDS or (tok == "만" and keep_man(tokens, i)):
            keep()
            i += 1
            continue
        month = next((m for m in MONTHS if tok.startswith(m) and TAIL_RE.fullmatch(tok[len(m):])), None)
        if month:
            out.append(MONTHS[month] + tok[len(month):])
            i += 1
            continue
        found = read_number(tokens, i)
        if not found:
            keep()
            i += 1
            continue
        num, suffix, used = found
        j = i + used
        nxt = tokens[j] if j < len(tokens) else ""
        after = tokens[j + 1] if j + 1 < len(tokens) else ""

        # 어림: "오 육 개월", "일 이 년", "이삼 일" 의 뒤쪽 수는 바꾸지 않는다
        if len(num) == 1 and num in DIGITS and prev and all(c in DIGITS for c in prev):
            keep()
            i += 1
            continue

        # 날짜·기간의 '일': "삼십일 치" -> 30일 치, "시월 이십일에" -> 10월 20일에
        if (len(num) >= 2 and num.endswith("일") and num[-2] in UNITS
                and not nxt.startswith("일")):
            if (DAY_MARK_RE.match(suffix) or (not suffix and DAY_MARK_RE.match(nxt) and nxt != "간")
                    or (MONTH_RE.search(prev) and value(num[:-1]) <= 31)):
                out.append(attach(value(num[:-1]), "일" + suffix))
                i = j
                continue
            if value(num[:-1]) <= 31 and (DAY_OR_NUMBER_RE.fullmatch(suffix)
                                          or (not suffix and nxt.startswith(("정도", "쯤")))):
                keep()  # "이십일에", "이십일 정도": 20일인지 21인지 모른다
                i += 1
                continue

        alt = round_alt(num, suffix, nxt)
        if alt:
            num, suffix = alt

        # 소수: "육 점 오", "영 점 이 오 밀리그램"
        if nxt in ("점", "쩜") and not suffix:
            digits, tail, k = read_fraction(tokens, j + 1)
            if digits:
                unit = tokens[k] if k < len(tokens) else ""
                if not tail and unit and not NOT_COUNTER_RE.match(unit) and COUNTER_TOKEN_RE.fullmatch(unit):
                    tail, k = unit, k + 1  # "영 점 오 밀리" -> 0.5밀리
                out.append(attach(f"{value(num)}.{digits}", tail))
                i = k
                continue
            if after and after[0] in DIGITS and not AFTER_POINT_OK_RE.match(after):
                keep()  # "삼 점 오르셨어요": 소수인지 점수인지 모른다
                i += 1
                continue

        # "통증이 칠 정도예요" -> 7 정도예요. "이 정도", "일 정도"(일, work)는 그대로
        if not suffix and nxt.startswith("정도") and num not in ("일", "이", "백", "천", "만"):
            out.append(str(value(num)))
            i = j
            continue

        counter = counter_of(num, nxt) if not suffix else None
        if counter:
            out.append(attach(value(num), week(nxt, after)))
            i = j + 1
            continue

        lexical = num in LEXICAL and not COUNTER_TOKEN_RE.fullmatch(suffix or "-")
        if lexical:
            keep()
            i += 1
            continue

        if (strong(num) or (alt and num in UNITS)
                or (suffix and ATTACHED_RE.fullmatch(suffix) and num + suffix[:1] != "이년")):
            out.append(attach(value(num), week(suffix, nxt)))
            i = j
            continue

        # 백·천 하나: 다른 수나 '이상' 앞에서만. "백에 육십" -> 100에 60 ('백'은 가방도 된다)
        if num in ("백", "천") and TAIL_RE.fullmatch(suffix):
            nf = read_number(tokens, j)
            if (nf and strong(nf[0])) or re.match(r"이상|이하|미만|초과|넘", nxt):
                out.append(attach(value(num), suffix))
                i = j
                continue

        keep()
        i += 1
    return " ".join(out)


CASES = [
    ("혈압이 백사십에 구십이에요", "혈압이 140에 90이에요"),
    ("이백 십 육이니까", "216이니까"),
    ("당화혈색소는 육 점 오예요", "당화혈색소는 6.5 예요"),
    ("당화혈색소가 육 점 오고요", "당화혈색소가 6.5 고요"),
    ("영 점 오 밀리", "0.5밀리"),
    ("영 점 이 오 밀리그램", "0.25밀리그램"),
    ("체온 삼십칠 점 오 도", "체온 37.5도"),
    ("삼 개월 치 드릴게요", "3개월 치 드릴게요"),
    ("오 밀리그램 드세요", "5밀리그램 드세요"),
    ("이 주 뒤에 오세요", "2주 뒤에 오세요"),
    ("삼십 분 정도", "30분 정도"),
    ("사 점 정도 아파요", "4점 정도 아파요"),
    ("이십대 여성", "20대 여성"),
    ("시월 이십 일에 오세요", "10월 20일에 오세요"),
    ("시월 이십일 일에", "10월 21일에"),
    ("시월 이십일에 오세요", "10월 20일에 오세요"),
    ("다음 달 이십일에 오세요", "다음 달 20일에 오세요"),
    ("이십일에 오세요", "이십일에 오세요"),
    ("이십일 정도 드세요", "이십일 정도 드세요"),
    ("혈당이 백이십일 정도", "혈당이 121 정도"),
    ("혈색소가 십일 점 오", "혈색소가 11.5"),
    ("칠 밀리미터", "7밀리미터"),
    ("당화혈색소 칠 점 이 퍼센트예요", "당화혈색소 7.2 퍼센트예요"),
    ("산소포화도 구십팔 퍼센트", "산소포화도 98 퍼센트"),
    ("삼 대 일", "삼 대 일"),
    ("삼십일 치 드릴게요", "30일 치 드릴게요"),
    ("구십일분 처방할게요", "90일분 처방할게요"),
    ("십일 뒤에 오세요", "10일 뒤에 오세요"),
    ("혈압이 백이십일에 팔십이에요", "혈압이 121에 80이에요"),
    ("일주일에 한 번", "1주일에 한 번"),
    ("일주일 뒤에 다시 오세요", "1주 뒤에 다시 오세요"),
    ("이 주일 후에 봬요", "2주 후에 봬요"),
    ("백사십이 넘으면", "140이 넘으면"),
    ("콜레스테롤이 이백 사십이고", "콜레스테롤이 240이고"),
    ("혈압이 백삼십에 팔십이세요", "혈압이 130에 80이세요"),
    ("백이 넘으면", "100이 넘으면"),
    ("백이에요", "100이에요"),
    ("백이십이랑", "120이랑"),
    ("십이 넘으면", "12 넘으면"),
    ("구십이예요", "92 예요"),
    ("수치가 백이십이고요", "수치가 120이고요"),
    ("오늘 혈압이 백에 육십이에요", "오늘 혈압이 100에 60이에요"),
    ("공복 혈당이 백 이십", "공복 혈당이 120"),
    ("공복 혈당 백 이상이면", "공복 혈당 100 이상이면"),
    ("만 이천 보", "12000 보"),
    ("만 원", "10000원"),
    ("만 육십오 세 이상", "만 65세 이상"),
    ("만 이 년 됐어요", "만 2년 됐어요"),
    ("삼십 점 만 점에 이십육 점", "30점 만 점에 26점"),
    ("이십 점 이상이면", "20점 이상이면"),
    ("유월에 뵐게요", "6월에 뵐게요"),
    ("이천 원이에요", "2000원이에요"),
    # 그대로 둬야 하는 것
    ("이 분은 괜찮아요", "이 분은 괜찮아요"),
    ("이 단계에서는 수술 안 해도 돼요", "이 단계에서는 수술 안 해도 돼요"),
    ("무슨 일 주로 하세요", "무슨 일 주로 하세요"),
    ("가벼운 일 정도는 괜찮아요", "가벼운 일 정도는 괜찮아요"),
    ("이 정도면 괜찮아요", "이 정도면 괜찮아요"),
    ("통증이 칠 정도예요", "통증이 7 정도예요"),
    ("일 시간 줄이세요", "일 시간 줄이세요"),
    ("오 육 개월 드세요", "오 육 개월 드세요"),
    ("이삼 일 정도 지켜보죠", "이삼 일 정도 지켜보죠"),
    ("일 이 년 됐어요", "일 이 년 됐어요"),
    ("삼 점 오르셨어요", "삼 점 오르셨어요"),
    ("만일 열이 나면 오세요", "만일 열이 나면 오세요"),
    ("아기가 백일 됐어요", "아기가 백일 됐어요"),
    ("이천에서 왔어요", "이천에서 왔어요"),
    ("천사 같아요", "천사 같아요"),
    ("백신 맞으셨죠", "백신 맞으셨죠"),
    ("백에 넣어 두세요", "백에 넣어 두세요"),
    ("천으로 감싸세요", "천으로 감싸세요"),
    ("천천히 드세요", "천천히 드세요"),
    ("사십견이 있어요", "사십견이 있어요"),
    ("십이지장 궤양이에요", "십이지장 궤양이에요"),
    ("오늘 사는 게 힘들어요", "오늘 사는 게 힘들어요"),
    ("이 점 이해해 주세요", "이 점 이해해 주세요"),
    ("천만에요", "천만에요"),
    ("팔이 아파요", "팔이 아파요"),
    ("이번에 오일 드세요", "이번에 오일 드세요"),
    ("만성 위염", "만성 위염"),
    ("이상 없어요", "이상 없어요"),
    ("", ""),
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
