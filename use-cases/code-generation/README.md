# Code Generation Fine-Tuning

Fine-tune CodeLlama-7B to generate clean, well-documented Python code with type hints.

## Quickstart

```bash
# Train (single 24GB GPU, ~60 min)
docker compose -f use-cases/code-generation/docker-compose.yml up train

# Merge & export GGUF
docker compose -f use-cases/code-generation/docker-compose.yml --profile export up merge

# Evaluate
docker compose -f use-cases/code-generation/docker-compose.yml --profile eval up eval
```

## Expected Results

| Metric | Value |
|--------|-------|
| **Training Time** | ~60 minutes (1x A10G 24GB) |
| **GPU Memory** | ~20 GB |
| **Estimated Cost** | $2.00 (A10G @ $0.50/hr) |
| **LoRA Rank** | 32 |
| **Seq Length** | 4096 |
| **Epochs** | 3 |

## Dataset

10 synthetic code generation tasks covering:
- CSV parsing with error handling
- LRU cache implementation
- Async HTTP fetching with semaphore
- Retry decorator with exponential backoff
- Timing context manager
- Dataclass with validation
- Nested dict flattening
- Singleton metaclass
- Fibonacci generator
- Merge sorted lists

All examples include type hints, docstrings, and follow Python best practices.

## Config Highlights

- **Base Model**: CodeLlama-7B-Instruct (code-specialized)
- **QLoRA 4-bit**: NF4 quantization, double quant
- **Higher Rank**: 32 for code complexity
- **Longer Context**: 4096 tokens for full functions
- **Lower LR**: 1e-4 for stable code training

## Output

- LoRA adapters: `use-cases/code-generation/out/`
- Merged model: `use-cases/code-generation/merged/`
- GGUF export: `use-cases/code-generation/merged/model-q4_k_m.gguf`

## Example Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("use-cases/code-generation/merged", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("use-cases/code-generation/merged")

prompt = "Write a function to validate email addresses using regex"
messages = [
    {"role": "system", "content": "You are a senior Python developer. Write clean, well-documented code with type hints."},
    {"role": "user", "content": prompt}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=512, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```