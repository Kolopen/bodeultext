#!/usr/bin/env python3
# Copyright 2026  Kolopen
# Apache 2.0
"""학습한 덧붙임 층(LoRA)을 whisper 에 합쳐 faster-whisper 형식으로 내보낸다.

  python local/whisper/export.py --lora exp/whisper/lora/final --out /mnt/e/voice-models/fw-aihub
  python local/whisper/export.py --out /mnt/e/voice-models/fw-turbo      # 학습 전 원본 (견주기용)

/mnt/e/voice-models/fw-이름 에 두면 local/eval_engines.sh 가 알아서 같이 잰다.
CPU 에서 쓰므로 int8 로 줄인다(약 0.8GB).
"""

import argparse
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="openai/whisper-large-v3-turbo")
    p.add_argument("--lora", type=Path, help="finetune.py 가 만든 폴더 (없으면 원본을 그대로 내보낸다)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--quantization", default="int8")
    args = p.parse_args()

    import torch
    from ctranslate2.converters import TransformersConverter
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base = args.base
    if args.lora:
        import json
        state = args.lora / "trainer_state.pt"
        if state.is_file():
            base = torch.load(state, map_location="cpu").get("base", base)
        cfg = json.loads((args.lora / "adapter_config.json").read_text())
        base = cfg.get("base_model_name_or_path") or base

    processor = WhisperProcessor.from_pretrained(args.lora or base)
    model = WhisperForConditionalGeneration.from_pretrained(base, torch_dtype=torch.float16)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora).merge_and_unload()
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        processor.save_pretrained(tmp)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        TransformersConverter(tmp, copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
            str(args.out), quantization=args.quantization, force=True)
    print(f"내보냈습니다: {args.out}")


if __name__ == "__main__":
    main()
