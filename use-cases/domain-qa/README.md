# Domain QA (Legal) Fine-Tuning

Fine-tune a 7B model to answer contract law questions accurately with legal reasoning.

## Quickstart

```bash
# Train (single 24GB GPU, ~45 min)
docker compose -f use-cases/domain-qa/docker-compose.yml up train

# Merge & export GGUF
docker compose -f use-cases/domain-qa/docker-compose.yml --profile export up merge

# Evaluate
docker compose -f use-cases/domain-qa/docker-compose.yml --profile eval up eval
```

## Expected Results

| Metric | Value |
|--------|-------|
| **Training Time** | ~45 minutes (1x A10G 24GB) |
| **GPU Memory** | ~18 GB |
| **Estimated Cost** | $1.50 (A10G @ $0.50/hr) |
| **LoRA Rank** | 16 |
| **Seq Length** | 2048 |
| **Epochs** | 3 |

## Dataset

15 Q&A pairs covering core contract law topics:
- Offer & acceptance
- Consideration
- Statute of frauds
- Parol evidence rule
- Duress & unconscionability
- Promissory estoppel
- Liquidated damages
- Assignment vs delegation
- Frustration/impossibility
- Anticipatory repudiation
- Third-party beneficiaries
- Mailbox rule
- Minor's capacity
- Breach remedies

## Config Highlights

- **Base Model**: Mistral-7B-Instruct-v0.2
- **QLoRA 4-bit**: NF4 quantization
- **Legal Domain**: Focused on contract law principles
- **Chat Format**: Structured Q&A with system prompt

## Output

- LoRA adapters: `use-cases/domain-qa/out/`
- Merged model: `use-cases/domain-qa/merged/`
- GGUF export: `use-cases/domain-qa/merged/model-q4_k_m.gguf`

## Example Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("use-cases/domain-qa/merged", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("use-cases/domain-qa/merged")

question = "What is promissory estoppel and when does it apply?"
messages = [
    {"role": "system", "content": "You are a legal assistant specializing in contract law. Provide accurate, concise answers citing relevant principles."},
    {"role": "user", "content": question}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=256, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```