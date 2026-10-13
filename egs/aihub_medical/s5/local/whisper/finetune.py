#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""whisper(large-v3-turbo)를 AI-Hub 의료 음성으로 추가 학습한다 (LoRA, GPU 8GB 용).

  python local/whisper/finetune.py --train exp/whisper/train.tsv --valid exp/whisper/valid.tsv \\
      --out exp/whisper/lora

모델 전체가 아니라 작은 덧붙임 층(LoRA)만 학습한다. 원래 whisper 가 가진 대화체·잡음에
대한 실력은 그대로 두고, 의료 용어와 이 데이터의 말투만 더 배운다. 그래서 8GB GPU 에도
들어가고, 결과물(덧붙임 층)도 수십 MB 다.

whisper 가 원래 배운 모양을 지킨다(그래야 faster-whisper·-AI 에서 그대로 돈다).
  - 이어 붙인 발화(prepare_data.py --pack-sec)는 구간마다 시각 표시를 단다.
      <|0.00|> 혈압이 140에 90이에요.<|3.22|><|3.22|> 약은 그대로 드세요.<|5.80|>
    faster-whisper 는 이 표시로 구간을 나누므로, 표시 없이만 배우면 그 쪽이 흔들린다.
    --timestamp-prob 의 비율만큼 표시를 달고 나머지는 표시 없이 배운다.
  - 소리의 일부를 가리는 SpecAugment 를 켠다. 같은 문장을 여러 사람이 읽은 데이터라
    문장을 외우지 않게 한다.

멈췄다 이어 하기: --save-every 걸음마다 out/last 에 저장한다. 같은 명령을 다시 돌리면
거기서 이어 간다(학습률·데이터 순서까지 그대로).

GPU 메모리가 모자라면(out of memory) --batch 4 --grad-accum 4 처럼 묶음을 줄인다.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare_asr import hangul  # noqa: E402

RATE = 16000
MAX_SAMPLES = 30 * RATE  # whisper 는 30초까지만 듣는다
MAX_LABEL = 448          # 디코더가 받는 최대 토큰 수


def read_tsv(path):
    """한 줄: 발화ID, wav 들, 길이들, 글들. 이어 붙인 발화는 '|' 로 잇는다."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            utt, wavs, durs, texts = line.rstrip("\n").split("\t", 3)
            rows.append((utt, wavs.split("|"), [float(d) for d in durs.split("|")], texts.split("|")))
    return rows


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != RATE:
        g = math.gcd(sr, RATE)
        audio = resample_poly(audio, RATE // g, sr // g).astype(np.float32)
    return audio


class Labels:
    """정답 글을 whisper 토큰으로 만든다. 시각 표시를 달거나 안 단다."""

    def __init__(self, tokenizer):
        ids = tokenizer.convert_tokens_to_ids
        self.tokenizer = tokenizer
        self.prefix = [ids("<|startoftranscript|>"), ids("<|ko|>"), ids("<|transcribe|>")]
        self.no_timestamps = ids("<|notimestamps|>")
        self.eot = ids("<|endoftext|>")
        self.ts0 = self.no_timestamps + 1  # <|0.00|>, 0.02초 간격으로 <|30.00|> 까지
        self.timestamps_ok = tokenizer.convert_ids_to_tokens(self.ts0) == "<|0.00|>"

    def time(self, sec):
        return self.ts0 + min(1500, max(0, round(sec / 0.02)))

    def text(self, s):
        return self.tokenizer.encode(" " + s.strip(), add_special_tokens=False)

    def __call__(self, pieces, timestamps):
        """pieces: [(시작 초, 끝 초, 글)]"""
        if timestamps and self.timestamps_ok:
            ids = list(self.prefix)
            for start, end, text in pieces:
                ids += [self.time(start)] + self.text(text) + [self.time(end)]
        else:
            ids = self.prefix + [self.no_timestamps] + self.text(" ".join(p[2] for p in pieces))
        return ids + [self.eot]


class Speech(Dataset):
    def __init__(self, rows, processor, timestamp_prob=0.0, seed=0):
        self.rows, self.processor = rows, processor
        self.labels = Labels(processor.tokenizer)
        self.timestamp_prob, self.seed = timestamp_prob, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        utt, wavs, _, texts = self.rows[i]
        audios = []
        for wav in wavs:
            a = load_audio(wav)
            if audios and sum(map(len, audios)) + len(a) > MAX_SAMPLES:
                break  # 30초를 넘는 뒤쪽 발화는 뺀다
            audios.append(a)
        # 줄마다 정해진 동전 던지기 (일꾼 프로세스가 몇이든, 이어 하든 같다)
        timestamps = random.Random(f"{self.seed}:{utt}").random() < self.timestamp_prob
        while True:
            pieces, start = [], 0
            for a, text in zip(audios, texts):
                pieces.append((start / RATE, (start + len(a)) / RATE, text))
                start += len(a)
            labels = self.labels(pieces, timestamps)
            if len(labels) <= MAX_LABEL or len(audios) == 1:
                break
            audios.pop()
        audio = np.concatenate(audios)[:MAX_SAMPLES]
        feats = self.processor.feature_extractor(audio, sampling_rate=RATE, return_tensors="np").input_features[0]
        return {"input_features": feats, "labels": labels[:MAX_LABEL],
                "text": " ".join(p[2] for p in pieces), "utt": utt}


def make_collate(decoder_start):
    def collate(batch):
        feats = torch.tensor(np.stack([b["input_features"] for b in batch]))
        longest = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
        # 모델이 디코더 입력을 만들 때 시작 토큰을 다시 붙이므로 정답에서는 뗀다.
        if (labels[:, 0] == decoder_start).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels,
                "texts": [b["text"] for b in batch], "utts": [b["utt"] for b in batch]}
    return collate


def cer(refs, hyps):
    """글자 오류율. 띄어쓰기·문장부호는 빼고 숫자는 한글로 읽어서 견준다(compare_asr.py 와 같게)."""
    errors = total = 0
    for r, h in zip(refs, hyps):
        r, h = hangul(r), hangul(h)
        prev = list(range(len(h) + 1))
        for i, x in enumerate(r, 1):
            cur = [i] + [0] * len(h)
            for j, y in enumerate(h, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
            prev = cur
        errors += prev[-1]
        total += len(r)
    return 100.0 * errors / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, processor, device, dtype, limit_batches=None):
    model.eval()
    refs, hyps = [], []
    for n, batch in enumerate(loader):
        if limit_batches and n >= limit_batches:
            break
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            out = model.generate(input_features=batch["input_features"].to(device, dtype),
                                 language="ko", task="transcribe", max_new_tokens=128, use_cache=True)
        hyps += processor.batch_decode(out, skip_special_tokens=True)
        refs += batch["texts"]
    model.train()
    return cer(refs, hyps), list(zip(refs[:3], hyps[:3]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="openai/whisper-large-v3-turbo", help="HuggingFace 이름 또는 폴더")
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--valid", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--timestamp-prob", type=float, default=0.5, help="시각 표시를 달고 배우는 비율")
    p.add_argument("--save-every", type=int, default=200, help="이 걸음마다 저장 (이어 하기용)")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=25, help="중간 평가에 쓸 묶음 수")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="시험용. GPU 없이 돌린다(아주 느림)")
    args = p.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type != "cuda" and not args.cpu:
        sys.exit("GPU 를 찾지 못했습니다. nvidia-smi 와 torch.cuda.is_available() 을 확인하세요.")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "cpu":
        dtype = torch.float32
    torch.manual_seed(args.seed)

    processor = WhisperProcessor.from_pretrained(args.base)
    processor.tokenizer.set_prefix_tokens(language="korean", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base, torch_dtype=dtype)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    model.config.apply_spec_augment = True  # 학습 때만 소리 일부를 가린다(기본 5%)
    # use_reentrant=False 여야 한다. 기본값(True)이면 인코더 첫 층(conv)이 얼어 있어
    # 인코더 쪽 덧붙임 층에 기울기가 안 흐를 수 있다.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    last = args.out / "last"
    if (last / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, last, is_trainable=True)
        print(f"이어서 학습합니다: {last}")
    else:
        lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none")
        model = get_peft_model(model, lora)
    for name, param in model.named_parameters():
        if param.requires_grad:  # 덧붙임 층은 float32 로 학습해야 안정적이다
            param.data = param.data.float()
    model.to(device)
    model.print_trainable_parameters()

    train_rows, valid_rows = read_tsv(args.train), read_tsv(args.valid)
    collate = make_collate(model.config.decoder_start_token_id)
    # 남는 묶음은 버리므로(drop_last) 내림으로 센다. 올림으로 세면 마지막 걸음을 영영 못 채운다.
    steps_per_epoch = max(1, len(train_rows) // args.batch // args.grad_accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    valid_loader = DataLoader(Speech(valid_rows, processor), batch_size=args.batch,
                              num_workers=args.workers, collate_fn=collate)
    if not Labels(processor.tokenizer).timestamps_ok:
        print("이 토크나이저에는 시각 표시 토큰이 없어 시각 표시 없이만 배웁니다.")

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / args.warmup) * max(0.0, 1 - s / total_steps))
    step = 0
    state_file = last / "trainer_state.pt"
    if state_file.is_file():
        state = torch.load(state_file, map_location="cpu")
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
    pieces = sum(len(r[1]) for r in train_rows)
    print(f"학습 {len(train_rows)}줄(발화 {pieces}개), 묶음 {args.batch}x{args.grad_accum}, "
          f"전체 {total_steps}걸음 (지금 {step}), 장치 {device} {dtype}")

    def save(where):
        where.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(where)
        processor.save_pretrained(where)
        torch.save({"optim": optim.state_dict(), "sched": sched.state_dict(), "step": step,
                    "base": args.base}, where / "trainer_state.pt")
        # torch 없이도 읽게 (local/stop_training.sh 가 몇 걸음까지 했는지 알려 준다)
        (where / "progress.json").write_text(json.dumps({"global_step": step, "total": total_steps}))

    model.train()
    started, done_since = time.time(), 0
    while step < total_steps:
        epoch = step // steps_per_epoch
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        skip = (step % steps_per_epoch) * args.batch * args.grad_accum  # 이어 할 때 본 데이터는 건너뛴다
        rows = [train_rows[i] for i in order[skip:]]
        loader = DataLoader(Speech(rows, processor, args.timestamp_prob, args.seed + epoch),
                            batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                            drop_last=True)
        if len(loader) < args.grad_accum:
            sys.exit(f"학습 줄이 너무 적습니다: {len(rows)}줄 (묶음 {args.batch}x{args.grad_accum})")
        accum, running = 0, 0.0
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                out = model(input_features=batch["input_features"].to(device, dtype),
                            labels=batch["labels"].to(device))
            (out.loss / args.grad_accum).backward()
            running += out.loss.item() / args.grad_accum
            accum += 1
            if accum < args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            accum, step, done_since = 0, step + 1, done_since + 1
            if step % 10 == 0 or step == total_steps:
                per = (time.time() - started) / done_since
                eta = (total_steps - step) * per / 3600
                print(f"걸음 {step}/{total_steps}  손실 {running:.3f}  학습률 {sched.get_last_lr()[0]:.2e}  "
                      f"{per:.1f}초/걸음  남은 시간 {eta:.1f}시간", flush=True)
            running = 0.0
            if step % args.save_every == 0 or step == total_steps:
                save(last)
            if step % args.eval_every == 0 or step == total_steps:
                score, examples = evaluate(model, valid_loader, processor, device, dtype, args.eval_batches)
                print(f"  평가: 글자 오류율 {score:.2f}%", flush=True)
                for r, h in examples:
                    print(f"    정답 {r}\n    결과 {h}", flush=True)
                with open(args.out / "eval.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "cer": score}, ensure_ascii=False) + "\n")
            if step >= total_steps or step % steps_per_epoch == 0:
                break  # 다 했거나 한 바퀴를 돌았다 (다음 바퀴는 순서를 새로 섞는다)
    save(args.out / "final")
    print(f"끝났습니다. 덧붙임 층: {args.out / 'final'}")


if __name__ == "__main__":
    main()
