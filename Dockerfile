FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command: run a sample training job
# Override with: docker run ... python src/train.py --config configs/medical_qa_qlora.yaml
CMD ["python", "-c", "import sys; print('LLM fine-tuning image ready. Override CMD to train.')"]
