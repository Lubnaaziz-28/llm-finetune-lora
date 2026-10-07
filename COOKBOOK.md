# Startup Fine-Tuning Cookbook

> **5 Use Cases, 5 Docker Commands** — Production-ready LoRA/QLoRA fine-tuning recipes for common startup AI needs.

Each use case includes: synthetic dataset, config, docker-compose, and expected cost/time on a single 24GB GPU.

---

## Quick Reference

| Use Case | Base Model | LoRA Rank | Seq Len | Epochs | Est. Time | Est. Cost |
|----------|------------|-----------|---------|--------|-----------|-----------|
| [Support Chatbot](#1-support-chatbot) | Mistral-7B-Instruct | 16 | 2048 | 3 | 45 min | $1.50 |
| [Code Generation](#2-code-generation) | CodeLlama-7B-Instruct | 32 | 4096 | 3 | 60 min | $2.00 |
| [Domain QA (Legal)](#3-domain-qa-legal) | Mistral-7B-Instruct | 16 | 2048 | 3 | 45 min | $1.50 |
| [Sentiment Analysis](#4-sentiment-analysis) | Mistral-7B-Instruct | 8 | 512 | 5 | 20 min | $0.70 |
| [Instruction Following](#5-instruction-following) | Mistral-7B-Instruct | 16 | 1024 | 3 | 35 min | $1.20 |

*Costs estimated on A10G 24GB @ $0.50/hr (RunPod/Lambda/AWS spot)*

---

## 1. Support Chatbot

**Goal**: Polite, concise customer support agent for a tech company.

### Commands

```bash
# Train
docker compose -f use-cases/support-chatbot/docker-compose.yml up train

# Merge & export GGUF (for llama.cpp)
docker compose -f use-cases/support-chatbot/docker-compose.yml --profile export up merge

# Evaluate
docker compose -f use-cases/support-chatbot/docker-compose.yml --profile eval up eval
```

### Expected Output

```
Training: 3 epochs, ~1500 steps
Loss curve: 2.1 → 1.2 → 0.9
GPU Memory: ~18 GB peak
Time: ~45 minutes
```

### Example Inference

```python
# Input: "My order #TC-8842 hasn't arrived yet."
# Output: "I apologize for the delay. Let me check tracking for #TC-8842 — it's out for delivery and should arrive by end of day. Want the tracking link?"
```

### Artifacts

- LoRA: `use-cases/support-chatbot/out/`
- Merged: `use-cases/support-chatbot/merged/`
- GGUF: `use-cases/support-chatbot/merged/model-q4_k_m.gguf`

---

## 2. Code Generation

**Goal**: Generate clean, typed Python code from natural language.

### Commands

```bash
# Train
docker compose -f use-cases/code-generation/docker-compose.yml up train

# Merge & export GGUF
docker compose -f use-cases/code-generation/docker-compose.yml --profile export up merge

# Evaluate
docker compose -f use-cases/code-generation/docker-compose.yml --profile eval up eval
```

### Expected Output

```
Training: 3 epochs, ~1000 steps
Loss curve: 2.3 → 1.4 → 1.0
GPU Memory: ~20 GB peak
Time: ~60 minutes
```

### Example Inference

```python
# Input: "Write an async function to fetch URLs with semaphore limit"
# Output: async def fetch_all(urls, max_concurrent=10):
#     semaphore = asyncio.Semaphore(max_concurrent)
#     async def fetch_one(session, url):
#         async with semaphore:
#             async with session.get(url) as r:
#                 return url, await r.text()
#     ...
```

### Artifacts

- LoRA: `use-cases/code-generation/out/`
- Merged: `use-cases/code-generation/merged/`
- GGUF: `use-cases/code-generation/merged/model-q4_k_m.gguf`

---

## 3. Domain QA (Legal)

**Goal**: Accurate contract law Q&A with legal reasoning.

### Commands

```bash
# Train
docker compose -f use-cases/domain-qa/docker-compose.yml up train

# Merge & export GGUF
docker compose -f use-cases/domain-qa/docker-compose.yml --profile export up merge

# Evaluate
docker compose -f use-cases/domain-qa/docker-compose.yml --profile eval up eval
```

### Expected Output

```
Training: 3 epochs, ~1500 steps
Loss curve: 2.0 → 1.1 → 0.8
GPU Memory: ~18 GB peak
Time: ~45 minutes
```

### Example Inference

```python
# Input: "What is promissory estoppel?"
# Output: "Promissory estoppel enforces a promise without consideration when: (1) clear promise, (2) reasonable reliance expected, (3) actual detrimental reliance, (4) enforcement needed to avoid injustice."
```

### Artifacts

- LoRA: `use-cases/domain-qa/out/`
- Merged: `use-cases/domain-qa/merged/`
- GGUF: `use-cases/domain-qa/merged/model-q4_k_m.gguf`

---

## 4. Sentiment Analysis

**Goal**: 3-class classification (Positive/Negative/Neutral) on short texts.

### Commands

```bash
# Train
docker compose -f use-cases/sentiment-analysis/docker-compose.yml up train

# Merge & export GGUF
docker compose -f use-cases/sentiment-analysis/docker-compose.yml --profile export up merge

# Evaluate
docker compose -f use-cases/sentiment-analysis/docker-compose.yml --profile eval up eval
```

### Expected Output

```
Training: 5 epochs, ~2000 steps
Loss curve: 1.8 → 0.9 → 0.5 → 0.3 → 0.2
GPU Memory: ~14 GB peak
Time: ~20 minutes
```

### Example Inference

```python
# Input: "This product is amazing! Best purchase ever."
# Output: "Positive"

# Input: "Save your money. This product is a scam."
# Output: "Negative"
```

### Artifacts

- LoRA: `use-cases/sentiment-analysis/out/`
- Merged: `use-cases/sentiment-analysis/merged/`
- GGUF: `use-cases/sentiment-analysis/merged/model-q4_k_m.gguf`

---

## 5. Instruction Following

**Goal**: Follow diverse formatting/transformation instructions precisely.

### Commands

```bash
# Train
docker compose -f use-cases/instruction-following/docker-compose.yml up train

# Merge & export GGUF
docker compose -f use-cases/instruction-following/docker-compose.yml --profile export up merge

# Evaluate
docker compose -f use-cases/instruction-following/docker-compose.yml --profile eval up eval
```

### Expected Output

```
Training: 3 epochs, ~1500 steps
Loss curve: 2.2 → 1.3 → 0.9
GPU Memory: ~18 GB peak
Time: ~35 minutes
```

### Example Inference

```python
# Input: "List 3 benefits of version control, one per line"
# Output: "Track changes over time\nCollaborate without conflicts\nRevert mistakes easily"

# Input: "Convert to JSON: name=John, age=30, city=NYC"
# Output: {"name": "John", "age": 30, "city": "NYC"}
```

### Artifacts

- LoRA: `use-cases/instruction-following/out/`
- Merged: `use-cases/instruction-following/merged/`
- GGUF: `use-cases/instruction-following/merged/model-q4_k_m.gguf`

---

## Validation

Run the validation script to check all configs:

```bash
python validate_configs.py
```

Expected output:
```
============================================================
Startup Fine-Tuning Cookbook - Config Validation
============================================================

Validating: support-chatbot
  ✅ Valid
Validating: code-generation
  ✅ Valid
Validating: domain-qa
  ✅ Valid
Validating: sentiment-analysis
  ✅ Valid
Validating: instruction-following
  ✅ Valid

============================================================
✅ All configs validated successfully!
```

---

## Requirements

- **GPU**: 24GB VRAM (A10G, A100, RTX 3090/4090)
- **Docker**: 20.10+ with NVIDIA Container Toolkit
- **Disk**: ~15 GB per use case (base model + checkpoints)

---

## Customization

### Change Base Model

Edit `model.name` in the config YAML:

```yaml
model:
  name: "meta-llama/Llama-2-13b-chat-hf"  # or any HF model
```

### Adjust LoRA Rank

```yaml
lora:
  r: 32        # higher = more capacity, more VRAM
  alpha: 64    # typically 2x r
```

### Change Sequence Length

```yaml
data:
  max_seq_length: 4096  # longer = more context, more VRAM
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps` |
| Slow Training | Ensure `bf16: true`, check GPU utilization with `nvidia-smi` |
| Config Validation Fails | Run `python validate_configs.py` for details |
| Model Not Found | Run `huggingface-cli login` or set `HF_TOKEN` env var |

---

## Next Steps

1. **Swap datasets** — Replace `dataset.jsonl` with your proprietary data
2. **Scale up** — Use `Llama-2-13B` or `Mistral-7B-v0.3` for better quality
3. **Serve** — Load GGUF with `llama.cpp` or merged model with `vLLM`/`TGI`
4. **Evaluate** — Add your test set to `data/eval_path` for proper metrics

---

*Part of the [LLM Fine-Tuning with LoRA/QLoRA](../README.md) pipeline.*