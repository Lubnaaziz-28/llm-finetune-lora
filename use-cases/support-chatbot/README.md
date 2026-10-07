# Support Chatbot Fine-Tuning

Fine-tune a 7B model to act as a polite, concise customer support agent for a tech company.

## Quickstart

```bash
# Train (single 24GB GPU, ~45 min)
docker compose -f use-cases/support-chatbot/docker-compose.yml up train

# Merge & export GGUF (for llama.cpp)
docker compose -f use-cases/support-chatbot/docker-compose.yml --profile export up merge

# Evaluate
docker compose -f use-cases/support-chatbot/docker-compose.yml --profile eval up eval
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

15 synthetic customer support conversations in chat format covering:
- Order tracking & delivery issues
- Returns & refunds
- Technical troubleshooting
- Account management
- Billing & subscriptions
- Security (2FA, password reset)

## Config Highlights

- **Base Model**: Mistral-7B-Instruct-v0.2 (already instruction-tuned)
- **QLoRA 4-bit**: NF4 quantization, double quant
- **Target Modules**: All attention + MLP layers
- **Learning Rate**: 2e-4 with cosine schedule
- **Batch Size**: 2 (effective 8 with grad accumulation)

## Output

- LoRA adapters: `use-cases/support-chatbot/out/`
- Merged model: `use-cases/support-chatbot/merged/`
- GGUF export: `use-cases/support-chatbot/merged/model-q4_k_m.gguf`

## Example Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("use-cases/support-chatbot/merged", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("use-cases/support-chatbot/merged")

messages = [
    {"role": "system", "content": "You are a helpful customer support agent for TechCorp."},
    {"role": "user", "content": "My order hasn't arrived yet."}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```