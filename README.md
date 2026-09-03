# llm-finetune-lora

Production-oriented LoRA/QLoRA fine-tuning pipeline for adapting open-weight LLMs (Llama, Mistral) to specialized corpora. Built for medical/legal/academic domain adaptation.

## Features
- 4-bit QLoRA training on a single 24GB GPU
- Config-driven (YAML): model, data, LoRA rank/alpha, schedule
- Built-in evaluation: perplexity + task metrics + hallucination spot-check
- Export to GGUF / merged HF format for low-latency serving

## Quickstart
```bash
pip install -r requirements.txt
python src/train.py --config configs/medical_qa_qlora.yaml
python src/merge_and_export.py --adapter out/adapter --base mistralai/Mistral-7B
```

## Results
| Setup | Base | LoRA r | Task metric |
|---|---|---|---|
| TBD | Mistral-7B | 16 | TBD |

## Contact
Dr. Lubna Aziz | engr.lubnaaziz@gmail.com | [Scholar](https://scholar.google.com/citations?user=Uu-CkiYAAAAJ)
