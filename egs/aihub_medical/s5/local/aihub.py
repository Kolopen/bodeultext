#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""AI-Hub '의료진 및 환자 음성' 라벨을 다룬다.

라벨은 음성 하나에 JSON 하나다. 이런 모양이다.

  {"전사정보": {"LabelText": "엄마 얼굴이 이상해지는 거 같아서요,"},
   "화자정보": {"Gender": "Female", "Age": "60~69", ...},
   "음성정보": {"SamplingRate": "48000", "NumberOfChannel": "1", ...},
   "파일정보": {"FileName": "PA_0322-2848-01-03-F-08-C.wav",
                "DirectoryPath": "/nia/metrixB/data/PA_0322", "FileLength": "2.34"}}

두 가지 일을 한다.

  inspect  라벨만 보고 전사 표기를 센다. 음성을 받기 전에 돌린다.
           숫자·영문·기호가 얼마나 섞였는지 알아야 정규화 규칙을 정할 수 있다.

  prep     라벨과 음성을 짝지어 Kaldi 데이터 폴더를 만든다.
           (wav.scp, text, utt2spk, spk2gender, utt2dur)

  python3 local/aihub.py inspect --labels /mnt/e/aihub/train/labels   (폴더, zip, 폴더 안의 zip)
  python3 local/aihub.py prep --labels .../labels/train --audio .../audio/train \\
      --out data/train
"""

import argparse
import collections
import json
import os
import re
import sys
import zipfile

# 비식별화 태그(#@이름# 등). 음성에는 실제 이름이 들어 있으므로 태그로
# 학습하면 소리와 글자가 어긋난다. 이런 발화는 뺀다.
DEID_RE = re.compile(r"#@[^#]*#")
# 이중전사 (숫자표기)/(한글표기). 소리를 학습하므로 뒤쪽을 쓴다.
DUAL_RE = re.compile(r"\(([^()]*)\)\s*/\s*\(([^()]*)\)")
# 간투어 뒤 슬래시("아/"), 영문 의학용어 표시("세츄레이션*")
FILLER_SLASH_RE = re.compile(r"(?<=\S)/")
# 소리가 없는 기호. 문장부호는 지운다.
PUNCT_RE = re.compile(r"[,.?!~…·\"'`“”‘’:;*]")
SPACE_RE = re.compile(r"\s+")
HANGUL_ONLY_RE = re.compile(r"^[가-힣 ]+$")


def normalize(text):
    """전사를 학습용 글자로 바꾼다. 쓸 수 없으면 (None, 이유)를 돌려준다."""
    if DEID_RE.search(text):
        return None, "비식별화 태그"
    text = DUAL_RE.sub(lambda m: m.group(2), text)
    text = FILLER_SLASH_RE.sub("", text)
    text = PUNCT_RE.sub(" ", text)
    text = text.replace("-", " ")
    text = SPACE_RE.sub(" ", text).strip()
    if not text:
        return None, "빈 전사"
    if re.search(r"[0-9]", text):
        return None, "숫자"
    if re.search(r"[A-Za-z]", text):
        return None, "영문"
    if not HANGUL_ONLY_RE.match(text):
        return None, "기타 기호"
    return text, None


def iter_zip_labels(path):
    # 라벨이 200만 개쯤이라 풀어 두면 옮기기만 몇 시간이다. zip 안에서 바로 읽는다.
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        sys.exit(f"zip 을 열 수 없습니다: {path}\n"
                 "  AI-Hub 가 큰 파일을 조각(.zip.part0, .zip.part1073741824 ...)으로 보냈다면\n"
                 "  조각을 순서대로 이어 붙인 zip 이어야 합니다. 조각이 있는 폴더에서:\n"
                 "    cat $(ls 라벨링데이터.zip.part* | sort -V) > 합친파일.zip")
    with archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".json"):
                continue
            where = f"{path}!{info.filename}"
            try:
                yield where, json.loads(archive.read(info).decode("utf-8-sig"))
            except (ValueError, UnicodeDecodeError) as e:
                print(f"읽을 수 없는 라벨: {where} ({e})", file=sys.stderr)


def iter_labels_raw(root):
    if os.path.isfile(root):
        yield from iter_zip_labels(root)
        return
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(dirpath, name)
            lower = name.lower()
            if lower.endswith(".zip"):
                yield from iter_zip_labels(path)
            elif lower.endswith(".json"):
                try:
                    with open(path, encoding="utf-8-sig") as f:
                        yield path, json.load(f)
                except (ValueError, UnicodeDecodeError) as e:
                    print(f"읽을 수 없는 라벨: {path} ({e})", file=sys.stderr)


def iter_labels(root):
    """폴더(하위 폴더까지), zip 파일, 폴더 안의 zip 을 모두 읽는다."""
    n = 0
    for item in iter_labels_raw(root):
        n += 1
        if n % 100000 == 0:
            # 수백만 개를 읽는 동안 멈춘 것처럼 보이지 않게 한다.
            print(f"  라벨 {n}개 읽는 중...", file=sys.stderr, flush=True)
        yield item


def field(label, *keys, default=""):
    cur = label
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def utt_id_of(label, path):
    name = field(label, "파일정보", "FileName") or os.path.basename(path)
    return os.path.splitext(os.path.basename(name))[0]


def speaker_of(utt_id):
    # PA_0322-2848-01-03-F-08-C -> PA_0322. Kaldi는 발화 ID가 화자 ID로
    # 시작해야 정렬이 맞는데, 이 이름 규칙이 마침 그렇다.
    return utt_id.split("-", 1)[0]


def seconds(label):
    try:
        return float(field(label, "파일정보", "FileLength", default="0"))
    except ValueError:
        return 0.0


def cmd_inspect(args):
    total = 0
    total_sec = 0.0
    kept_sec = 0.0
    reasons = collections.Counter()
    examples = collections.defaultdict(list)
    chars = collections.Counter()
    patterns = collections.Counter()
    pattern_examples = collections.defaultdict(list)
    meta = {k: collections.Counter() for k in
            ("화자 구분", "성별", "나이", "녹음 환경", "녹음 기기", "샘플레이트", "채널")}
    speakers = set()
    sentences = collections.Counter()

    checks = {
        "숫자": re.compile(r"[0-9]"),
        "영문": re.compile(r"[A-Za-z]"),
        "괄호": re.compile(r"[()]"),
        "슬래시": re.compile(r"/"),
        "#태그": re.compile(r"#"),
        "*표시": re.compile(r"\*"),
    }

    for path, label in iter_labels(args.labels):
        total += 1
        text = str(field(label, "전사정보", "LabelText"))
        sec = seconds(label)
        total_sec += sec
        utt = utt_id_of(label, path)
        speakers.add(speaker_of(utt))
        meta["화자 구분"][utt[:2]] += 1
        meta["성별"][field(label, "화자정보", "Gender")] += 1
        meta["나이"][field(label, "화자정보", "Age")] += 1
        meta["녹음 환경"][field(label, "환경정보", "RecordingEnviron")] += 1
        meta["녹음 기기"][field(label, "환경정보", "RecordingDevice")] += 1
        meta["샘플레이트"][field(label, "음성정보", "SamplingRate")] += 1
        meta["채널"][field(label, "음성정보", "NumberOfChannel")] += 1

        for ch in text:
            if not ("가" <= ch <= "힣" or ch == " "):
                chars[ch] += 1
        for name, regex in checks.items():
            if regex.search(text):
                patterns[name] += 1
                if len(pattern_examples[name]) < args.examples:
                    pattern_examples[name].append(text)

        norm, reason = normalize(text)
        if norm is None:
            reasons[reason] += 1
            if len(examples[reason]) < args.examples:
                examples[reason].append(text)
        else:
            kept_sec += sec
            sentences[norm] += 1

    if total == 0:
        sys.exit(f"라벨(.json)을 못 찾았습니다: {args.labels}")

    def pct(n, d):
        return f"{100.0 * n / d:.1f}%" if d else "-"

    print(f"라벨 {total}개, {total_sec / 3600:.1f}시간, 화자 {len(speakers)}명")
    kept = total - sum(reasons.values())
    print(f"학습에 쓸 수 있는 발화 {kept}개 ({pct(kept, total)}), {kept_sec / 3600:.1f}시간")
    repeated = sum(c for c in sentences.values() if c > 1)
    print(f"서로 다른 문장 {len(sentences)}개, 여러 번 읽힌 문장에 속한 발화 {pct(repeated, kept)}")
    print()
    print("[빠지는 이유]")
    for reason, n in reasons.most_common():
        print(f"  {reason:10s} {n:8d}  {pct(n, total)}")
        for ex in examples[reason]:
            print(f"      {ex}")
    print()
    print("[전사에 섞인 표기]")
    for name in checks:
        n = patterns[name]
        print(f"  {name:10s} {n:8d}  {pct(n, total)}")
        for ex in pattern_examples[name]:
            print(f"      {ex}")
    print()
    print("[한글·공백 외 글자] 많은 순 30개")
    print("  " + "  ".join(f"{repr(c)}:{n}" for c, n in chars.most_common(30)))
    print()
    for name, counter in meta.items():
        print(f"[{name}] " + ", ".join(f"{k or '(없음)'}:{v}" for k, v in counter.most_common(12)))


def index_audio(root):
    found = {}
    dup = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(".wav"):
                continue
            key = os.path.splitext(name)[0]
            if key in found:
                dup += 1
                continue
            found[key] = os.path.join(dirpath, name)
    if dup:
        print(f"같은 이름의 음성 {dup}개는 처음 것만 씁니다.", file=sys.stderr)
    return found


def cmd_prep(args):
    audio = index_audio(args.audio)
    if not audio:
        sys.exit(f"음성(.wav)을 못 찾았습니다: {args.audio}")

    rows = []
    missing = 0
    reasons = collections.Counter()
    for path, label in iter_labels(args.labels):
        utt = utt_id_of(label, path)
        wav = audio.get(utt)
        if wav is None:
            # 음성을 일부만 받았으면 나머지 라벨은 짝이 없다. 정상이다.
            missing += 1
            continue
        if re.search(r"\s", wav):
            sys.exit(f"경로에 공백이 있으면 Kaldi가 못 읽습니다. 폴더 이름을 바꿔 주세요: {wav}")
        text, reason = normalize(str(field(label, "전사정보", "LabelText")))
        if text is None:
            reasons[reason] += 1
            continue
        sec = seconds(label)
        if sec and (sec < args.min_sec or sec > args.max_sec):
            reasons["길이"] += 1
            continue
        rate = str(field(label, "음성정보", "SamplingRate"))
        gender = "f" if str(field(label, "화자정보", "Gender")).lower().startswith("f") else "m"
        rows.append((utt, speaker_of(utt), wav, rate, text, gender, sec))

    if not rows:
        sys.exit("짝지어진 발화가 없습니다. --labels 와 --audio 가 같은 세트(Training/Validation)인지 확인해 주세요.")

    rows.sort(key=lambda r: r[0])
    if args.max_utts and len(rows) > args.max_utts:
        # 화자가 고르게 들어가도록 일정 간격으로 고른다.
        step = len(rows) / args.max_utts
        rows = [rows[int(i * step)] for i in range(args.max_utts)]

    os.makedirs(args.out, exist_ok=True)

    def write(name, lines):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def wav_entry(wav, rate):
        if rate == str(args.rate):
            return wav
        # 48kHz 원본을 16kHz로 내려서 읽는다. 원본은 건드리지 않는다.
        return f"sox {wav} -t wav -r {args.rate} -b 16 -c 1 - |"

    write("wav.scp", (f"{u} {wav_entry(w, r)}" for u, _, w, r, _, _, _ in rows))
    write("text", (f"{u} {t}" for u, _, _, _, t, _, _ in rows))
    write("utt2spk", (f"{u} {s}" for u, s, _, _, _, _, _ in rows))
    if all(r[6] for r in rows):
        write("utt2dur", (f"{u} {d:.2f}" for u, _, _, _, _, _, d in rows))
    genders = {}
    for _, s, _, _, _, g, _ in rows:
        genders.setdefault(s, g)
    write("spk2gender", (f"{s} {g}" for s, g in sorted(genders.items())))

    hours = sum(r[6] for r in rows) / 3600
    print(f"{args.out}: 발화 {len(rows)}개, {hours:.1f}시간, 화자 {len(genders)}명")
    if missing:
        print(f"  음성이 없는 라벨 {missing}개 (아직 안 받은 음성)")
    for reason, n in reasons.most_common():
        print(f"  뺀 발화: {reason} {n}개")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="라벨의 전사 표기를 센다")
    p.add_argument("--labels", required=True, help="라벨 폴더(하위 폴더까지) 또는 라벨 zip")
    p.add_argument("--examples", type=int, default=5, help="항목마다 보여줄 예시 수")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("prep", help="Kaldi 데이터 폴더를 만든다")
    p.add_argument("--labels", required=True, help="라벨 폴더(하위 폴더까지) 또는 라벨 zip")
    p.add_argument("--audio", required=True, help="원천데이터(.wav) 폴더")
    p.add_argument("--out", required=True)
    p.add_argument("--rate", type=int, default=16000)
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-sec", type=float, default=20.0)
    p.add_argument("--max-utts", type=int, default=0, help="0이면 전부")
    p.set_defaults(func=cmd_prep)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
