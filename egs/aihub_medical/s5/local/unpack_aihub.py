#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""AI-Hub 에서 받은 파일에서 음성(wav)을 바로 꺼낸다.

AI-Hub 다운로드는 이렇게 겹겹이 싸여 있다.

  받은 파일 (tar. 브라우저가 이름을 '환자1' 처럼 바꾸기도 한다)
   └─ 141.의료진_및_환자_음성/.../원천데이터/
       ├─ 환자_1.zip.part0               진짜 zip 의 1GB 조각
       ├─ 환자_1.zip.part1073741824      조각 이름 끝의 숫자는 zip 안에서의 위치
       └─ ...

보통은 tar 를 풀고, 조각을 이어 붙이고, 그 zip 을 다시 푼다. 21GB 짜리면 같은
크기의 임시 파일이 두 번 생기고 시간도 그만큼 든다. 여기서는 조각들을 이어진
zip 하나처럼 읽어서 wav 만 바로 꺼낸다.

  python3 local/unpack_aihub.py 받은파일 --out /mnt/e/aihub/train/audio
  python3 local/unpack_aihub.py 조각들이있는폴더 --out ...
  python3 local/unpack_aihub.py 합친.zip --out ...

라벨은 풀 필요가 없다(local/aihub.py 가 zip 안을 바로 읽는다). 조각을 이어 붙인
zip 하나만 만들려면 --join 을 쓴다.

  python3 local/unpack_aihub.py 받은라벨파일 --join /mnt/e/aihub/valid/labels

이미 같은 크기로 꺼낸 파일은 건너뛰므로 중간에 멈췄으면 같은 명령을 다시 하면 된다.
파일마다 CRC 를 확인하고, 깨진 파일은 남기지 않고 개수만 알려준다.
"""

import argparse
import os
import re
import sys
import tarfile
import zipfile

PART_RE = re.compile(r"^(?P<base>.+\.zip)\.part(?P<offset>\d+)$", re.IGNORECASE)


class Segments:
    """여러 파일 조각(경로, 시작 위치, 길이)을 이어진 파일 하나처럼 읽는다."""

    def __init__(self, segments):
        self.segments = []
        start = 0
        for path, offset, size in segments:
            self.segments.append((start, path, offset, size))
            start += size
        self.size = start
        self.pos = 0
        self.handles = {}

    def _handle(self, path):
        if path not in self.handles:
            self.handles[path] = open(path, "rb")
        return self.handles[path]

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, pos, whence=0):
        if whence == 1:
            pos += self.pos
        elif whence == 2:
            pos += self.size
        self.pos = max(0, pos)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = []
        while n > 0 and self.pos < self.size:
            for start, path, offset, size in self.segments:
                if start <= self.pos < start + size:
                    break
            within = self.pos - start
            take = min(n, size - within)
            f = self._handle(path)
            f.seek(offset + within)
            data = f.read(take)
            if not data:
                raise IOError(f"조각을 끝까지 읽지 못했습니다: {path}")
            out.append(data)
            self.pos += len(data)
            n -= len(data)
        return b"".join(out)

    def close(self):
        for f in self.handles.values():
            f.close()
        self.handles = {}


def group_parts(entries):
    """(이름, 경로, 시작, 길이) 목록에서 zip 별로 조각을 순서대로 묶는다."""
    zips = {}
    for name, path, offset, size in entries:
        base = os.path.basename(name)
        m = PART_RE.match(base)
        if m:
            zips.setdefault(m.group("base"), []).append((int(m.group("offset")), path, offset, size))
        elif base.lower().endswith(".zip"):
            zips.setdefault(base, []).append((0, path, offset, size))
    groups = {}
    for base, parts in zips.items():
        parts.sort()
        expected = 0
        for at, path, offset, size in parts:
            if at != expected:
                sys.exit(f"{base}: 조각이 빠졌습니다 ({expected} 위치의 조각이 없음). 다시 받아 주세요.")
            expected += size
        groups[base] = [(path, offset, size) for _, path, offset, size in parts]
    return groups


def find_zips(src):
    if os.path.isdir(src):
        entries = []
        for dirpath, _, files in os.walk(src):
            for name in files:
                path = os.path.join(dirpath, name)
                entries.append((name, path, 0, os.path.getsize(path)))
        return group_parts(entries)
    if zipfile.is_zipfile(src):
        return {os.path.basename(src): [(src, 0, os.path.getsize(src))]}
    if tarfile.is_tarfile(src):
        entries = []
        total = os.path.getsize(src)
        cut = ("받은 파일이 끝까지 다운로드되지 않았습니다: " + src +
               "\n  브라우저 다운로드 목록(Ctrl+J)에서 이어 받거나 다시 받아 주세요.")
        try:
            with tarfile.open(src) as tar:
                for member in tar:
                    if member.isfile():
                        if member.offset_data + member.size > total:
                            sys.exit(cut)
                        entries.append((member.name, src, member.offset_data, member.size))
        except (tarfile.ReadError, EOFError):
            sys.exit(cut)
        return group_parts(entries)
    sys.exit(f"tar, zip, 조각 폴더 중 어느 것도 아닙니다: {src}")


def extract(name, segments, out, ext):
    stream = Segments(segments)
    try:
        archive = zipfile.ZipFile(stream)
    except zipfile.BadZipFile:
        sys.exit(f"{name}: zip 으로 열 수 없습니다. 다운로드가 끝까지 됐는지 확인해 주세요.")
    members = [i for i in archive.infolist()
               if not i.is_dir() and i.filename.lower().endswith(ext)]
    print(f"{name}: {len(members)}개 ({stream.size / 1e9:.1f}GB) -> {out}", flush=True)
    done = skipped = bad = 0
    for k, info in enumerate(members, 1):
        base = os.path.basename(info.filename)
        # 화자별 폴더에 둔다 (PA_0046-596-...wav -> PA_0046/)
        target = os.path.join(out, base.split("-", 1)[0], base)
        if os.path.exists(target) and os.path.getsize(target) == info.file_size:
            skipped += 1
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            tmp = target + ".tmp"
            try:
                with archive.open(info) as src, open(tmp, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                os.replace(tmp, target)
                done += 1
            except (zipfile.BadZipFile, OSError) as e:
                bad += 1
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"  깨진 파일: {info.filename} ({e})", file=sys.stderr)
        if k % 5000 == 0:
            print(f"  {k}/{len(members)}", flush=True)
    archive.close()
    stream.close()
    print(f"{name}: 새로 꺼냄 {done}, 이미 있어 건너뜀 {skipped}, 깨짐 {bad}")
    return bad


def join(name, segments, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, name)
    stream = Segments(segments)
    tmp = target + ".tmp"
    with open(tmp, "wb") as dst:
        while True:
            chunk = stream.read(1 << 24)
            if not chunk:
                break
            dst.write(chunk)
    stream.close()
    os.replace(tmp, target)
    if not zipfile.is_zipfile(target):
        sys.exit(f"{target}: 이어 붙였지만 zip 으로 열리지 않습니다. 다시 받아 주세요.")
    print(f"{target} ({stream.size / 1e9:.2f}GB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", nargs="+", help="받은 파일(tar), 조각 폴더, 또는 zip")
    parser.add_argument("--out", help="wav 를 꺼낼 폴더 (공백 없는 경로)")
    parser.add_argument("--join", metavar="DIR",
                        help="꺼내지 않고 조각을 이어 붙인 zip 을 이 폴더에 만든다 (라벨용)")
    parser.add_argument("--ext", default=".wav", help="꺼낼 파일 확장자")
    args = parser.parse_args()

    if not args.out and not args.join:
        sys.exit("--out (wav 꺼내기) 이나 --join (zip 만들기) 중 하나를 주세요.")
    if args.out and re.search(r"\s", os.path.abspath(args.out)):
        sys.exit("--out 경로에 공백이 있으면 Kaldi 가 못 읽습니다.")
    bad = 0
    for src in args.src:
        groups = find_zips(src)
        if not groups:
            sys.exit(f"zip 이나 zip 조각을 찾지 못했습니다: {src}\n"
                     "  다운로드가 덜 됐거나 다른 파일일 수 있습니다.")
        if args.join:
            for name, segments in sorted(groups.items()):
                join(name, segments, args.join)
            continue
        for name, segments in sorted(groups.items()):
            bad += extract(name, segments, args.out, args.ext.lower())
    if bad:
        sys.exit(f"깨진 파일 {bad}개. 같은 명령을 다시 하면 그것만 다시 시도합니다. "
                 "계속 깨지면 다시 받아야 합니다.")


if __name__ == "__main__":
    main()
