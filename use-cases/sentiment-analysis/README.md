# Sentiment Analysis Fine-Tuning

Fine-tune a 7B model for 3-class sentiment classification (Positive/Negative/Neutral).

## Quickstart

```bash
# Train (single 24GB GPU, ~20 min)
docker compose -f use-cases/sentiment-analysis/docker-compose.yml up train

# Merge & export GGUF
docker compose -f use-cases/sentiment-analysis/docker-compose.yml --profile export up merge

# Evaluate
docker compose -f use-cases/sentiment-analysis/docker-compose.yml --profile eval up eval
```

## Expected Results

| Metric | Value |
|--------|-------|
| **Training Time** | ~20 minutes (1x A10G 24GB) |
| **GPU Memory** | ~14 GB |
| **Estimated Cost** | $0.70 (A10G @ $0.50/hr) |
| **LoRA Rank** | 8 |
| **Seq Length** | 512 |
| **Epochs** | 5 |

## Dataset

20 labeled examples for 3-class classification:
- Positive: 7 examples (enthusiastic reviews)
- Negative: 7 examples (complaints, dissatisfaction)
- Neutral: 6 examples (factual, mixed, indifferent)

Short texts (reviews, feedback) with single-label outputs.

## Config Highlights

- **Base Model**: Mistral-7B-Instruct-v0.2
- **Lower Rank**: 8 (simpler classification task)
- **Shorter Context**: 512 tokens (short inputs)
- **Higher LR**: 3e-4 (faster convergence for classification)
- **More Epochs**: 5 (small dataset)
- **Target Modules**: Attention only (no MLP)

## Output

- LoRA adapters: `use-cases/sentiment-analysis/out/`
- Merged model: `use-cases/sentiment-analysis/merged/`
- GGUF export: `use-cases/sentiment-analysis/merged/model-q4_k_m.gguf`

## Example Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("use-cases/sentiment-analysis/merged", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("use-cases/sentiment-analysis/merged")

text = "This product is amazing! Best purchase ever."
messages = [
    {"role": "system", "content": "Classify the sentiment of the text as Positive, Negative, or Neutral. Respond with only the label."},
    {"role": "user", "content": text}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=16, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split()[-1])
# Output: Positive
```