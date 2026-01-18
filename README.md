# Qwen LLM Fine-Tuning with LoRA / QLoRA  
**Production-Oriented Parameter-Efficient Fine-Tuning**

## 📌 Project Overview
This project demonstrates **parameter-efficient fine-tuning (PEFT)** of a large language model using **LoRA / QLoRA**, focusing on **efficient training, scalability, and production-ready inference**.

The goal is to adapt a pretrained **Qwen2.5-1.5B-Instruct** model to domain-specific instructions while minimizing GPU memory usage and deployment cost, following best practices used in real-world LLM systems.

---

## 🚀 Key Objectives
- Fine-tune a large language model efficiently without full weight updates  
- Apply **4-bit quantization** to enable training on limited GPU resources  
- Design instruction-style datasets suitable for supervised fine-tuning  
- Deploy the fine-tuned model using **vLLM** with runtime LoRA injection  
- Build an end-to-end **training → evaluation → inference pipeline**

---

## 🧠 Core Concepts Covered
- Parameter-Efficient Fine-Tuning (PEFT)  
- Low-Rank Adaptation (LoRA)  
- QLoRA (Quantized LoRA)  
- Instruction Tuning  
- LLM Quantization (4-bit)  
- Scalable Inference with vLLM  

---

## 🏗️ System Architecture
```
Dataset (JSONL, Instruction Format)
        ↓
Tokenizer & Preprocessing
        ↓
Base Model (Qwen2.5-1.5B-Instruct)
        ↓
4-bit Quantization (BitsAndBytes)
        ↓
LoRA / QLoRA Adapters (PEFT)
        ↓
Supervised Fine-Tuning
        ↓
Evaluation
        ↓
vLLM Inference with Runtime LoRA Injection
```

---

## 📂 Dataset Design
The dataset is formatted in **instruction-tuning style (JSONL)** with explicit role separation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain overfitting in machine learning."},
    {"role": "assistant", "content": "Overfitting occurs when a model learns noise instead of patterns..."}
  ]
}
```

### Dataset Characteristics
- System / User / Assistant roles  
- Compatible with instruction-tuned LLMs  
- Easily extendable to multi-turn conversations  

---

## ⚙️ Training Setup

### Base Model
- **Qwen2.5-1.5B-Instruct**

### Fine-Tuning Strategy
- **LoRA / QLoRA**
- Rank (r): `16`
- Alpha (α): `32`
- Target Modules: Attention projection layers (`q_proj`, `v_proj`)
- Bias: `none`

### Quantization
- **4-bit quantization** using `bitsandbytes`  
- Enables training on limited VRAM while maintaining performance  

### Frameworks
- HuggingFace `Transformers`  
- `PEFT` for adapter-based fine-tuning  
- `BitsAndBytes` for quantization  

---

## 🧪 Training & Evaluation
- Supervised fine-tuning (SFT) on instruction-style data  
- Loss monitoring during training  
- Qualitative evaluation via prompt-based testing  
- Comparison between base vs fine-tuned responses  

---

## 🚀 Inference & Deployment

### vLLM Inference
The fine-tuned LoRA adapters are served using **vLLM**, allowing:
- High-throughput inference  
- Runtime LoRA injection (no model duplication)  
- Cost-efficient deployment  

Example:
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --enable-lora \
  --lora-modules my_adapter=./lora_adapter
```

---

## 📈 Results & Outcomes
- Successfully adapted a large LLM with **minimal trainable parameters**  
- Reduced GPU memory usage via 4-bit quantization  
- Achieved efficient, scalable inference suitable for production systems  
- Demonstrated a full understanding of **LLM fine-tuning trade-offs and deployment pipelines**  

---

## 🛠️ Tech Stack
- **Python**
- **HuggingFace Transformers**
- **PEFT (LoRA / QLoRA)**
- **BitsAndBytes**
- **vLLM**
- **PyTorch**

---

## 🎯 Use Cases
- Domain-specific chatbots  
- Enterprise knowledge assistants  
- LLMs integrated with RAG systems  
- Cost-efficient LLM deployment  

---

## 📌 Future Improvements
- Add automatic evaluation metrics (BLEU / ROUGE / GPT-based eval)  
- Integrate with a Retrieval-Augmented Generation (RAG) pipeline  
- Deploy as a FastAPI service  
- Experiment with multi-LoRA adapters for multi-domain tuning  

---

## 👤 Author
**Omar Fadlallah**  
Machine Learning Engineer | AI Engineer (NLP & RAG Systems)

---

## ⭐ Final Note
This project reflects **real-world LLM fine-tuning practices**, focusing on **efficiency, scalability, and production readiness** rather than experimental toy examples.
