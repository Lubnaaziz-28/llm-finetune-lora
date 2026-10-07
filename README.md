<div align="center">

# LLM Fine-Tuning with LoRA/QLoRA

### Domain-Adapt Open-Weight LLMs on a Single GPU

[![CI](https://img.shields.io/github/actions/workflow/status/Lubnaaziz-28/llm-finetune-lora/ci.yml?logo=github&style=flat-square)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)]()
[![Tech](https://img.shields.io/badge/Tech-LoRA%2FQLoRA-F1C40F)]()
[![Scale](https://img.shields.io/badge/Scale-7B_to_70B-2ECC71)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)]()
[![FineTuneBench](https://img.shields.io/badge/FineTuneBench-70B_%24.4.20-green?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDQwIDQwIj48cGF0aCBkPSJNMTAgMTBoNHY0SDEweiIvPjwvc3ZnPg==)](benchmarks/index.html)

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
| QLoRA | Mistral-7B | 16 | Hallucination -40% | 1x 24GB |
| QLoRA | LLaMA-2-13B | 32 | Hallucination -40% | 1x 24GB |

## FineTuneBench

Live cost-vs-quality leaderboard for QLoRA fine-tuning. Results are from actual runs on an NVIDIA A100 40GB at $0.80/hr.

| Rank | Model | Dataset | Hours | GPU Cost | Eval Score | Hallucination Rate |
|---|---|---|---|---|---|---|
| 1 | Qwen-2.5-7B-Instruct | medical_qa | 6.2h | $4.96 | 0.841 | 0.131 |
| 2 | LLaMA-3-8B-Instruct | medical_qa | 7.4h | $5.92 | 0.834 | 0.138 |
| 3 | Mistral-7B-v0.3 | medical_qa | 6.8h | $5.44 | 0.812 | 0.162 |

> **Cost-per-run tip:** QLoRA r=16 with identical configs keeps training memory under 20GB, leaving headroom for larger batch sizes on a single A100.

**Leaderboard:** [View live →](benchmarks/index.html)

**Reproduce:**

```bash
# Run benchmark
python scripts/benchmark_runner.py --config configs/qlora_benchmark.yaml

# Generate markdown + HTML leaderboard
python scripts/generate_leaderboard.py
```

Raw results CSV: [`benchmarks/results.csv`](benchmarks/results.csv)

## Citation

```bibtex
@software{aziz2024lorafinetune,
  title={LoRA/QLoRA Fine-Tuning Pipeline},
  author={Aziz, Lubna},
  year={2024}
}
```

## Startup Cookbook

**5 Use Cases, 5 Docker Commands** — Ready-to-run fine-tuning recipes for common startup AI needs.

| Use Case | Description | Base Model | Est. Time | Est. Cost |
|----------|-------------|------------|-----------|-----------|
| [Support Chatbot](use-cases/support-chatbot) | Polite customer support agent | Mistral-7B-Instruct | 45 min | $1.50 |
| [Code Generation](use-cases/code-generation) | Clean typed Python from NL | CodeLlama-7B-Instruct | 60 min | $2.00 |
| [Domain QA (Legal)](use-cases/domain-qa) | Contract law Q&A | Mistral-7B-Instruct | 45 min | $1.50 |
| [Sentiment Analysis](use-cases/sentiment-analysis) | 3-class classification | Mistral-7B-Instruct | 20 min | $0.70 |
| [Instruction Following](use-cases/instruction-following) | Formatting/transform tasks | Mistral-7B-Instruct | 35 min | $1.20 |

### Quickstart

```bash
# Pick a use case and run:
docker compose -f use-cases/support-chatbot/docker-compose.yml up train

# Merge & export GGUF for serving:
docker compose -f use-cases/support-chatbot/docker-compose.yml --profile export up merge
```

### Full Documentation

See [COOKBOOK.md](COOKBOOK.md) for detailed commands, expected outputs, and customization guide.

### Validate All Configs

```bash
python validate_configs.py
```

## Contact

Dr. Lubna Aziz, engr.lubnaaziz@gmail.com, [Google Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)
