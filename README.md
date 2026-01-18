Qwen LLM Fine-Tuning — LoRA / QLoRA (Production-Oriented)

Independent Project

Fine-tuned Qwen2.5-1.5B-Instruct using parameter-efficient LoRA / QLoRA, enabling domain adaptation with minimal GPU memory footprint.
Designed and formatted instruction-style datasets (JSONL) for supervised fine-tuning, including system, user, and assistant roles.
Applied 4-bit quantization (bitsandbytes) and configured LoRA adapters (r=16, α=32) targeting attention projection layers to optimize training efficiency.
Trained and evaluated the model using Transformers + PEFT, then deployed it for inference using vLLM with runtime LoRA injection, enabling scalable and cost-efficient serving.
Demonstrated strong understanding of LLM fine-tuning, optimization trade-offs, and production inference pipelines.

Tech: Python, Transformers, PEFT, LoRA, QLoRA, BitsAndBytes, vLLM, HuggingFace
