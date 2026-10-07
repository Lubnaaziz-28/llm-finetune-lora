# Instruction Following Fine-Tuning

Fine-tune a 7B model to follow diverse formatting and transformation instructions precisely.

## Quickstart

```bash
# Train (single 24GB GPU, ~35 min)
docker compose -f use-cases/instruction-following/docker-compose.yml up train

# Merge & export GGUF
docker compose -f use-cases/instruction-following/docker-compose.yml --profile export up merge

# Evaluate
docker compose -f use-cases/instruction-following/docker-compose.yml --profile eval up eval
```

## Expected Results

| Metric | Value |
|--------|-------|
| **Training Time** | ~35 minutes (1x A10G 24GB) |
| **GPU Memory** | ~18 GB |
| **Estimated Cost** | $1.20 (A10G @ $0.50/hr) |
| **LoRA Rank** | 16 |
| **Seq Length** | 1024 |
| **Epochs** | 3 |

## Dataset

20 diverse instruction-following tasks:
- Creative writing (haiku)
- Structured output (lists, tables, JSON, YAML)
- Code generation (one-liners, regex, SQL)
- Text transformation (case, voice, reverse)
- Extraction (emails, tags)
- Formatting (markdown, commit messages)
- Translation
- Generation (UUID, passwords)
- Counting/arithmetic

## Config Highlights

- **Base Model**: Mistral-7B-Instruct-v0.2
- **QLoRA 4-bit**: NF4 quantization
- **Medium Context**: 1024 tokens (varied output lengths)
- **Full Module Targeting**: Attention + MLP for diverse tasks

## Output

- LoRA adapters: `use-cases/instruction-following/out/`
- Merged model: `use-cases/instruction-following/merged/`
- GGUF export: `use-cases/instruction-following/merged/model-q4_k_m.gguf`

## Example Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("use-cases/instruction-following/merged", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("use-cases/instruction-following/merged")

instruction = "Write a Python one-liner to flatten a list of lists"
messages = [
    {"role": "system", "content": "Follow the instruction precisely. Output only the requested format."},
    {"role": "user", "content": instruction}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=128, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```