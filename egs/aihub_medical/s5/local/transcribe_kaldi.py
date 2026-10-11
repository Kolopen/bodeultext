#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""진료 녹음을 학습한 Kaldi 모델로 받아 적는다.

결과는 -AI 의 voice-transcribe 와 같은 모양(JSON)이라 voice-analyze 에 그대로
넣으면 리포트가 나온다. 보통은 local/report_audio.sh 가 이걸 부른다.

  python3 local/transcribe_kaldi.py 녹음.m4a --out 녹음.json
  python3 local/transcribe_kaldi.py 녹음.m4a --out 녹음.json \\
      --segmentation 모델/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \\
      --embedding 모델/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \\
      --speakers 2

순서
  1. 녹음을 16kHz 모노로 읽는다 (m4a, mp3, wav 다 된다. -AI 의 load_audio)
  2. 자른다. 화자분리 모델을 주면 사람별 발언으로(-AI 의 diarize),
     VAD 만 주면 말소리 구간으로 먼저 나누고, 그 안을 다시 쉬는 틈(--pause)에서
     끊는다. 학습 데이터가 한 문장씩이라 문장 길이로 넣어야 잘 맞힌다.
     화자분리를 쓰면 쉬는 틈에서 잡음을 재서 걷어낸다(-AI 의 reduce_noise).
  3. 구간마다 Kaldi 로 인식한다 (online2-wav-nnet3-latgen-faster).
     같은 사람의 구간은 목소리 특징(i-vector)을 이어 받아 갈수록 잘 맞춘다.
  4. 조각("+이")을 어절로 다시 붙이고, 한글 수를 숫자로 바꾼다(local/korean_itn.py).
     raw_text 에는 바꾸기 전 글을 남긴다.

학습한 데이터가 낭독체라 실제 진료 대화에서는 틀리는 곳이 더 많다. 리포트의
근거 문장과 시각([00:09])을 보고 녹음을 다시 들어 확인한다.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy 가 없습니다. -AI 를 먼저 설치하세요 (다음단계.txt 의 [6-1]):\n"
             "  pip3 install --break-system-packages -e ~/-AI[transcribe]")

HERE = Path(__file__).resolve().parent
S5 = HERE.parent
KALDI_ROOT = S5.parents[2]
sys.path.insert(0, str(HERE))
from korean_itn import convert as itn  # noqa: E402


def import_voice_ai():
    """-AI 의 전사 도구를 불러온다. 설치(pip -e)를 안 했으면 ~/-AI/src 에서 찾는다."""
    try:
        from voice_ai import transcribe
    except ImportError:
        for root in (Path.home() / "-AI" / "src", KALDI_ROOT.parent / "-AI" / "src"):
            if (root / "voice_ai").is_dir():
                sys.path.insert(0, str(root))
                break
        try:
            from voice_ai import transcribe
        except ImportError as e:
            sys.exit(f"-AI 의 voice_ai 를 불러올 수 없습니다 ({e}).\n"
                     "  pip3 install --break-system-packages -e ~/-AI[transcribe]\n"
                     "  (다음단계.txt 의 [6-1] 참고)")
    return transcribe


def kaldi_env():
    env = dict(os.environ)
    bins = [KALDI_ROOT / "src" / d for d in ("online2bin", "latbin", "bin", "fstbin")]
    bins.append(KALDI_ROOT / "tools" / "openfst" / "bin")
    env["PATH"] = os.pathsep.join(map(str, bins)) + os.pathsep + env.get("PATH", "")
    return env


def default_graph():
    for g in ("exp/chain/tree/graph_big", "exp/chain/tree/graph"):
        if (S5 / g / "HCLG.fst").is_file():
            return S5 / g
    return S5 / "exp/chain/tree/graph"


def default_lmwt(graph):
    """평가 때 가장 좋았던 언어모델 가중치. 그래프에 맞는 평가 결과를 찾는다.

    그래프 폴더 안의 best_cer 를 먼저 본다. 모델을 다른 곳으로 옮길 때 같이 복사해
    두면(다음단계.txt [10]) 거기서도 같은 값을 쓴다.
    """
    decode = "decode_test_big" if graph.name == "graph_big" else "decode_test"
    scoring = S5 / "exp/chain/tdnn1a" / decode / "scoring_kaldi"
    for best in (graph / "best_cer", graph / "best_wer", scoring / "best_cer", scoring / "best_wer"):
        if best.is_file():
            found = re.search(r"_(\d+)_([\d.]+)\s*$", best.read_text().strip())
            if found:
                return int(found.group(1)), float(found.group(2))
    return 10, 0.0


def portable_config(model, tmp):
    """online.conf 에 적힌 절대 경로를 지금 모델 폴더로 고쳐 쓴다.

    prepare_online_decoding.sh 는 만든 자리의 경로를 적는다. 모델 폴더를 E: 나
    다른 컴퓨터로 옮기면 그대로는 못 읽는다.
    """
    def fix(line):
        key, eq, val = line.strip().partition("=")
        for sub in ("conf", "ivector_extractor"):
            mark = f"/{sub}/"
            if eq and mark in val:
                return f"{key}={model / sub / val.rsplit(mark, 1)[1]}"
        return line.strip()

    ivec = tmp / "ivector_extractor.conf"
    ivec.write_text("\n".join(fix(l) for l in (model / "conf/ivector_extractor.conf").read_text().splitlines()) + "\n")
    lines = []
    for line in (model / "conf/online.conf").read_text().splitlines():
        line = fix(line)
        if line.startswith("--ivector-extraction-config="):
            line = f"--ivector-extraction-config={ivec}"
        lines.append(line)
    online = tmp / "online.conf"
    online.write_text("\n".join(lines) + "\n")
    return online


def vad_turns(audio, vad_model, rate, max_speech):
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(vad_model)
    config.silero_vad.min_silence_duration = 0.25
    config.silero_vad.max_speech_duration = max_speech
    config.sample_rate = rate
    window = config.silero_vad.window_size
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
    turns = []

    def drain():
        while not vad.empty():
            seg = vad.front
            start = round(seg.start * 1000 / rate)
            turns.append(("", start, start + round(len(seg.samples) * 1000 / rate)))
            vad.pop()

    for offset in range(0, len(audio), window):
        block = audio[offset:offset + window]
        if len(block) < window:
            block = np.pad(block, (0, window - len(block)))
        vad.accept_waveform(block)
        drain()
    vad.flush()
    drain()
    return turns


def split_pauses(samples, rate, min_pause, keep=0.3):
    """쉬는 틈에서 끊어 (시작, 끝) 목록을 돌려준다. 모델 없이 소리 크기로만 본다.

    학습 데이터가 한 문장씩 읽은 녹음이라 언어모델도 문장 단위로 배웠다. 여러
    문장을 한 덩어리로 넣거나 문장 한가운데를 자르면 그 자리에서 틀린다.
    쉬는 틈은 양쪽에 keep 초씩 남기고 그보다 긴 가운데만 건너뛴다(긴 무음에서는
    모델이 없는 말을 지어내기 쉽다). 잡음이 큰 녹음에서는 작게 말한 사람의 소리가
    쉬는 틈으로 보일 수 있으므로 기준을 낮게 잡고, 짧은 틈은 버리지 않는다.
    """
    hop, win = rate // 100, rate * 25 // 1000
    n = (len(samples) - win) // hop
    if min_pause <= 0 or n < 50:
        return [(0, len(samples))]
    power = np.array([np.mean(samples[i * hop:i * hop + win] ** 2) for i in range(n)])
    db = 10 * np.log10(power + 1e-10)
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    # 잡음이 커서 말소리와 차이가 작을수록 기준을 낮춰 덜 끊는다.
    speech = db > floor + min(6.0, 0.3 * (peak - floor))
    if not speech.any():
        return [(0, len(samples))]
    need, k = int(min_pause * 100), int(keep * 100)
    voiced = np.flatnonzero(speech)
    first, last = int(voiced[0]), int(voiced[-1]) + 1
    spans, start, quiet_from = [], max(0, first - k), None
    for i in range(first, last):
        if not speech[i]:
            if quiet_from is None:
                quiet_from = i
            continue
        if quiet_from is not None and i - quiet_from >= need:
            if i - quiet_from <= 2 * k:  # 짧은 틈은 한가운데서 끊는다
                mid = (quiet_from + i) // 2
                spans.append((start, mid))
                start = mid
            else:
                spans.append((start, quiet_from + k))
                start = i - k
        quiet_from = None
    end = len(samples) if last + k >= n else (last + k) * hop + win
    bounds = [(a * hop, b * hop) for a, b in spans] + [(start * hop, end)]
    if first - k <= 0:
        bounds[0] = (0, bounds[0][1])
    return bounds


def write_wav(path, samples, rate):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


# 평가(steps/nnet3/decode.sh --post-decode-acwt 10)와 같은 눈금으로 맞춘다.
# 그래야 평가에서 고른 언어모델 가중치(LMWT)를 그대로 쓸 수 있다.
POST_DECODE_ACWT = 10.0


def decode(utts, args, tmp, env):
    """utts: [(utt_id, spk_id, wav_path)] -> {utt_id: [단어 번호]}"""
    model, graph = args.model, args.graph
    online = portable_config(model, tmp)
    fsf = (model / "frame_subsampling_factor").read_text().strip() \
        if (model / "frame_subsampling_factor").is_file() else "1"
    lmwt, wip = args.lmwt
    q = shlex.quote
    weighting, online_mode = "", "false"
    if args.silence_weight < 1.0:
        # 쉬는 소리는 목소리 특징(i-vector)을 잴 때 덜 센다. Kaldi 기본은 끔(1.0).
        # 어디가 쉬는 소리인지는 인식이 진행되며 알게 되므로 조금씩(--online=true)
        # 넣어야 한다. 한 번에 넣으면 모든 소리가 쉬는 소리로 셈해진다.
        sil = (graph / "phones/silence.csl").read_text().strip()
        weighting = (f"--ivector-silence-weighting.silence-weight={args.silence_weight} "
                     f"--ivector-silence-weighting.silence-phones={sil} "
                     f"--ivector-silence-weighting.max-state-duration=40 ")
        online_mode = "true"
    nj = max(1, min(args.nj, len(utts)))
    jobs = []
    for j in range(nj):
        part = utts[j * len(utts) // nj:(j + 1) * len(utts) // nj]
        d = tmp / f"job{j}"
        d.mkdir()
        (d / "wav.scp").write_text("".join(f"{u} {p}\n" for u, _, p in part))
        log = q(str(d / "decode.log"))
        spk2utt = {}
        for u, s, _ in part:
            spk2utt.setdefault(s, []).append(u)
        (d / "spk2utt").write_text("".join(f"{s} {' '.join(us)}\n" for s, us in sorted(spk2utt.items())))
        cmd = (
            f"online2-wav-nnet3-latgen-faster --online={online_mode} --do-endpointing=false "
            f"--frame-subsampling-factor={fsf} {q('--config=' + str(online))} {weighting}"
            f"--max-active={args.max_active} --beam={args.beam} --lattice-beam=6.0 "
            f"--acoustic-scale=1.0 {q(str(model / 'final.mdl'))} {q(str(graph / 'HCLG.fst'))} "
            f"{q(f'ark:{d}/spk2utt')} {q(f'scp:{d}/wav.scp')} ark:- 2> {log} | "
            f"lattice-scale --acoustic-scale={POST_DECODE_ACWT / lmwt} ark:- ark:- 2>> {log} | "
            f"lattice-add-penalty --word-ins-penalty={wip} ark:- ark:- 2>> {log} | "
            f"lattice-best-path ark:- {q(f'ark,t:{d}/words.int')} 2>> {log}"
        )
        jobs.append((d, subprocess.Popen(["bash", "-c", "set -o pipefail; " + cmd], env=env)))
    result = {}
    for d, proc in jobs:
        if proc.wait() != 0:
            path = d / "decode.log"
            log = path.read_text(errors="replace") if path.is_file() else "(로그가 없습니다)"
            errors = "\n".join(l for l in log.splitlines() if "ERROR" in l or "Usage" in l)
            sys.exit(f"Kaldi 인식이 실패했습니다.\n{errors[:1500]}\n로그 끝부분:\n{log[-1500:]}")
        for line in (d / "words.int").read_text().splitlines():
            parts = line.split()
            if parts:
                result[parts[0]] = parts[1:]
    return result


def to_text(ids, words):
    """단어 번호 -> 어절. "+이" 같은 조각은 앞 어절에 붙인다."""
    out = []
    for i in ids:
        w = words.get(i, "")
        if not w or w in ("<UNK>", "!SIL", "<eps>"):
            continue
        if w.startswith("+") and out:
            out[-1] += w[1:]
        else:
            out.append(w.lstrip("+"))
    return " ".join(out)


def stamp(ms):
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path, help="녹음 파일 (m4a, mp3, wav ...)")
    p.add_argument("--out", type=Path, help="결과 JSON (기본: 녹음 이름.json)")
    p.add_argument("--model", type=Path, default=S5 / "exp/chain/tdnn1a_online",
                   help="prepare_online_decoding 이 만든 모델 폴더")
    p.add_argument("--graph", type=Path, help="HCLG.fst 가 든 폴더 (기본: graph_big 이 있으면 그것)")
    p.add_argument("--lmwt", type=float, help="언어모델 가중치 (기본: 평가에서 가장 좋았던 값)")
    p.add_argument("--segmentation", type=Path, help="화자분리 모델 (pyannote segmentation 3-0)")
    p.add_argument("--embedding", type=Path, help="화자 임베딩 모델 (3dspeaker eres2net)")
    p.add_argument("--speakers", type=int, default=-1, help="화자 수를 알면 (의사+환자면 2)")
    p.add_argument("--manager", type=Path, help="voice-enroll 로 만든 매니저 성문(.json)")
    p.add_argument("--vad", type=Path, help="silero_vad.onnx (화자분리 없이 말소리 구간만 자를 때)")
    p.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=True,
                   help="화자분리를 쓸 때 쉬는 틈의 잡음을 걷어낸다")
    p.add_argument("--pause", type=float, default=0.4,
                   help="이만큼(초) 조용하면 끊는다. 0 이면 쉬는 틈을 보지 않는다")
    p.add_argument("--max-chunk", type=float, default=15.0, help="구간 최대 길이(초). 넘으면 조용한 지점에서 끊는다")
    p.add_argument("--min-chunk", type=float, default=0.3, help="이보다 짧은 구간은 버린다(초)")
    p.add_argument("--silence-weight", type=float, default=1.0,
                   help="i-vector 를 잴 때 쉬는 소리의 무게 (1.0 은 끔, 0.001 이 흔한 값). 잡음이 많을 때 시험해 본다")
    p.add_argument("--beam", type=float, default=15.0)
    p.add_argument("--max-active", type=int, default=7000)
    p.add_argument("--nj", type=int, default=4, help="동시에 돌릴 인식 작업 수")
    p.add_argument("--threads", type=int, default=4, help="화자분리 스레드 수")
    p.add_argument("--keep", type=Path, help="중간 파일(구간 wav, 로그)을 남길 폴더")
    args = p.parse_args()

    if not args.audio.is_file():
        p.error(f"녹음 파일이 없습니다: {args.audio}")
    if args.segmentation and not args.embedding:
        p.error("--segmentation 을 쓰려면 --embedding 도 필요합니다.")
    if args.manager and not args.segmentation:
        p.error("--manager 는 화자분리(--segmentation, --embedding)와 함께 씁니다.")
    args.graph = args.graph or default_graph()
    tuned = default_lmwt(args.graph)  # --lmwt 를 줘도 평가에서 고른 단어 삽입 벌점은 그대로 쓴다
    args.lmwt = (args.lmwt, tuned[1]) if args.lmwt else tuned
    args.out = args.out or args.audio.with_suffix(".json")
    for f in (args.model / "final.mdl", args.model / "conf/online.conf", args.graph / "HCLG.fst",
              args.graph / "words.txt"):
        if not f.is_file():
            sys.exit(f"없음: {f}\n  신경망 학습(run_tdnn.sh 15단계까지)이 끝났는지, --model/--graph 가 맞는지 보세요.")
    env = kaldi_env()
    if not shutil.which("online2-wav-nnet3-latgen-faster", path=env["PATH"]):
        sys.exit(f"Kaldi 프로그램을 찾을 수 없습니다 ({KALDI_ROOT}/src). Kaldi 빌드를 확인하세요.")

    vt = import_voice_ai()
    rate = vt.SAMPLE_RATE
    audio = vt.load_audio(args.audio)
    print(f"오디오 {len(audio) / rate:.1f}초를 읽었습니다.")

    if args.segmentation or args.vad:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            sys.exit("화자분리·VAD 에 쓰는 sherpa-onnx 가 없습니다 (다음단계.txt 의 [6-1]):\n"
                     "  pip3 install --break-system-packages -e ~/-AI[transcribe]")
    if args.segmentation:
        print("화자를 나누는 중입니다...")
        turns = vt.diarize(audio, segmentation_model=args.segmentation, embedding_model=args.embedding,
                           num_speakers=args.speakers, num_threads=args.threads)
        if args.denoise:
            profile = vt.noise_profile(audio, turns)
            if profile is not None:
                before = vt.noise_floor(audio, turns)
                audio = vt.reduce_noise(audio, profile)
                print(f"잡음 {before:.4f} -> {vt.noise_floor(audio, turns):.4f} 로 걷었습니다.")
        print(f"화자 {len({s for s, _, _ in turns})}명, 발언 {len(turns)}구간")
    elif args.vad:
        turns = vad_turns(audio, args.vad, rate, args.max_chunk)
        print(f"말소리 {len(turns)}구간")
    else:
        turns = [("", 0, round(len(audio) * 1000 / rate))]
        print("화자분리 없이 쉬는 틈에서 자릅니다 (화자는 리포트 단계가 문장 말투로 가른다).")

    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="kaldi-asr-", dir=args.keep))
    try:
        pieces = []
        max_samples, min_samples = int(args.max_chunk * rate), int(args.min_chunk * rate)
        for speaker, start_ms, end_ms in turns:
            seg = audio[int(start_ms * rate / 1000):int(end_ms * rate / 1000)]
            for a, b in split_pauses(seg, rate, args.pause):
                for offset, part in vt._split_long(seg[a:b], max_samples):
                    if len(part) < min_samples:
                        continue
                    start = start_ms + round((a + offset) * 1000 / rate)
                    pieces.append((speaker, start, start + round(len(part) * 1000 / rate), part))

        utts = []
        for n, (speaker, start, end, part) in enumerate(pieces):
            # 같은 화자의 구간은 목소리 특징을 이어 받는다. 화자를 모르면 구간마다 따로 본다.
            spk = re.sub(r"\W", "_", speaker) if speaker else f"u{n:05d}"
            utt = f"{spk}-{n:05d}"
            path = tmp / f"{utt}.wav"
            write_wav(path, part, rate)
            utts.append((utt, spk, path))
        utts.sort()
        print(f"{len(utts)}구간을 인식합니다 (모델 {args.model}, 그래프 {args.graph.name}, "
              f"언어모델 가중치 {args.lmwt[0]:g})...")
        hyps = decode(utts, args, tmp, env) if utts else {}

        words = {}
        for line in (args.graph / "words.txt").read_text(encoding="utf-8").splitlines():
            w, i = line.split()
            words[i] = w
        chunks = []
        empty = 0
        for n, (speaker, start, end, _) in enumerate(pieces):
            spk = re.sub(r"\W", "_", speaker) if speaker else f"u{n:05d}"
            raw = to_text(hyps.get(f"{spk}-{n:05d}", []), words)
            if not raw:
                empty += 1
                continue
            chunks.append({"index": len(chunks), "speaker": speaker, "start_ms": start,
                           "end_ms": end, "raw_text": raw, "text": itn(raw)})
        if empty:
            print(f"{empty}구간은 받아 적은 말이 없어 뺐습니다 (쉬는 소리, 기침, 잡음 등).")
        if args.manager:
            chunks = vt._mark_manager(audio, turns, chunks, args.manager, args.embedding, args.threads)
    finally:
        if args.keep:
            print(f"중간 파일: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    payload = {"engine": "kaldi", "model": str(args.model), "graph": str(args.graph), "chunks": chunks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    txt = args.out.with_suffix(".txt")
    txt.write_text("".join(f"[{stamp(c['start_ms'])}] {c['speaker'] or '-'}: {c['text']}\n" for c in chunks),
                   encoding="utf-8")
    if not chunks:
        print("한 구간도 받아 적지 못했습니다. 녹음이 비었는지, 모델이 맞는지 보세요.", file=sys.stderr)
        return 1
    print(f"{len(chunks)}구간 -> {args.out} (사람이 읽는 판: {txt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
