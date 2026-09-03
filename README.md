<div align="center">

# LLM Fine-Tuning with LoRA/QLoRA

### Domain-Adapt Open-Weight LLMs on a Single GPU

[![Tech](https://img.shields.io/badge/Tech-LoRA%2FQLoRA-F1C40F)]()
[![Scale](https://img.shields.io/badge/Scale-7B_to_70B-2ECC71)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)]()

*Production-oriented fine-tuning pipeline for medical, legal, and academic domain adaptation.*

</div>

---

## The Problem

General-purpose LLMs lack domain expertise. Fine-tuning a 7B+ parameter model typically requires expensive multi-GPU setups and complex configurations.

## The Solution

**QLoRA** (Quantized LoRA) enables fine-tuning 7B-70B parameter models on a **single 24GB GPU** with minimal quality loss.

```
┌─────────────────────────────────────────────────┐
│                Base Model (4-bit)                │
│              (Llama / Mistral / etc.)            │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              LoRA Adapters (16-bit)              │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│    │  Query   │  │  Value   │  │  Output  │    │
│    │  Layer   │  │  Layer   │  │  Layer   │    │
│    └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│         Merged Model (for serving)               │
│    → GGUF export for llama.cpp                   │
│    → HF format for vLLM / TGI                    │
└─────────────────────────────────────────────────┘
```

## Features

- **4-bit QLoRA** training on a single 24GB GPU
- **Config-driven** (YAML): model, data, LoRA rank/alpha, schedule
- **Evaluation**: perplexity + task metrics + hallucination spot-check
- **Export**: GGUF / merged HF format for low-latency serving

## Quickstart

```bash
pip install -r requirements.txt

# Train
python src/train.py --config configs/medical_qa_qlora.yaml

# Merge & export
python src/merge_and_export.py --adapter out/adapter --base mistralai/Mistral-7B

# Evaluate
python src/evaluate.py --model merged/ --eval-data data/eval.jsonl
```

## Results

| Setup | Base Model | LoRA Rank | Task Metric | GPU |
|---|---|---|---|---|
| QLoRA | Mistral-7B | 16 | TBD | 1x 24GB |
| QLoRA | Llama-2-13B | 32 | TBD | 1x 24GB |

## Citation

```bibtex
@software{aziz2024lorafinetune,
  title={LoRA/QLoRA Fine-Tuning Pipeline},
  author={Aziz, Lubna},
  year={2024}
}
```

## Contact

Dr. Lubna Aziz — engr.lubnaaziz@gmail.com — [Google Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)
