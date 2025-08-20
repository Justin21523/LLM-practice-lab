# 🤖 AI LLM Practice Labs

> **Comprehensive Hands-on Learning Project for Large Language Models (LLMs)** – A complete practical guide covering LLM × RAG × Agent × Fine-tuning from fundamentals to production deployment.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36%2B-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Project Overview

This project contains **30+ Jupyter Notebooks** that provide a complete learning path from PyTorch foundations to production-level LLM application deployment. It focuses on:

- 🧠 **LLM Applications**
- 🔍 **Retrieval-Augmented Generation (RAG)**
- 🤖 **AI Agents**
- 🎯 **Fine-tuning techniques**
- 🚀 **Production Deployment**

## 🎯 Learning Objectives

- ✅ Master the modern LLM development technology stack
- ✅ Build RAG-based knowledge Q&A systems (with Chinese-first optimization)
- ✅ Construct intelligent multi-agent applications
- ✅ Learn efficient fine-tuning and evaluation methods
- ✅ Gain production-level AI service deployment skills

## 📁 Project Structure

```

ai-llm-practice/
├── 📚 notebooks/                    # Main learning content (30+ notebooks)
│   ├── part\_a\_foundations/          # Part A: PyTorch Foundations (5 notebooks)
│   ├── part\_b\_transformer\_hf/       # Part B: Transformer & HF (4 notebooks)
│   ├── part\_c\_llm\_applications/     # Part C: LLM Applications (10 notebooks)
│   ├── part\_d\_finetuning/           # Part D: Fine-tuning (6 notebooks)
│   ├── part\_e\_rag\_agents/           # Part E: RAG × Agent (5 notebooks)
│   └── part\_f\_webui\_api/            # Part F: WebUI & API Deployment (2 notebooks)
├── 🛠️ shared\_utils/                 # Shared utility modules
├── ⚙️ configs/                      # Configuration files
├── 🧪 tests/                        # Test suite
├── 📊 monitoring/                   # Monitoring configuration
├── 🐳 deployment/                   # Deployment scripts
└── 📋 docs/                         # Documentation

````

## 🗂️ Full Curriculum

### 📘 Part A: Foundations – 5 notebooks

| Notebook                              | Topic                            | Core Skills                          | Status |
| ------------------------------------- | -------------------------------- | ------------------------------------ | ------ |
| `nb01_tensor_autograd.ipynb`          | Tensor Operations & Autograd     | PyTorch basics, gradient computation | ✅      |
| `nb02_nn_module_training.ipynb`       | Custom nn.Module & Training Loop | Model definition, training process   | ✅      |
| `nb03_data_preprocessing.ipynb`       | HF Datasets Preprocessing        | Data processing, pipeline building   | ✅      |
| `nb04_cnn_image_classification.ipynb` | CNN Image Classification         | Convolutional networks, image tasks  | ✅      |
| `nb05_lstm_text_generation.ipynb`     | LSTM/GRU Text Generation         | Sequence models, text generation     | ✅      |

### 🔧 Part B: Transformer & HF – 4 notebooks

| Notebook                           | Topic                            | Core Skills                                    | Status |
| ---------------------------------- | -------------------------------- | ---------------------------------------------- | ------ |
| `nb06_attention_transformer.ipynb` | Attention & Transformer          | Attention mechanisms, Transformer architecture | ✅      |
| `nb07_hf_datasets_pipeline.ipynb`  | HF Datasets Pipeline             | Multimodal data processing                     | ✅      |
| `nb08_hf_models_loading.ipynb`     | Model Loading & Inference        | Using pre-trained models                       | ✅      |
| `nb09_generation_strategies.ipynb` | Generation Strategies & Decoding | Top-k/p, temperature, beam search              | ✅      |

### 🚀 Part C: LLM Applications – 10 notebooks

| Notebook                               | Topic                             | Core Skills                                   | Status |
| -------------------------------------- | --------------------------------- | --------------------------------------------- | ------ |
| `nb10_text_generation_gptqwen.ipynb`   | GPT/Qwen/DeepSeek Text Generation | LLM-based text generation                     | ✅      |
| `nb11_instruction_tuning_demo.ipynb`   | Instruction Tuning Demo           | Instruction following, SFT                    | ✅      |
| `nb12_llm_evaluation_metrics.ipynb`    | LLM Evaluation Metrics            | Model evaluation, benchmarks                  | ✅      |
| `nb13_function_calling_tools.ipynb`    | **🔥 Function Calling & Tools**    | **Tool calling, LangChain**                   | ✅      |
| `nb14_react_multistep_reasoning.ipynb` | ReAct Multi-step Reasoning        | Reasoning chains, thought processes           | ✅      |
| `nb15_code_assistant_agent.ipynb`      | Code Assistant Agent              | Code generation, debugging                    | ✅      |
| `nb16_document_ie_extraction.ipynb`    | Document Information Extraction   | IE, structured extraction                     | ✅      |
| `nb17_multilingual_translation.ipynb`  | Multilingual Translation          | Multilingual processing, translation          | ✅      |
| `nb18_safety_alignment_redteam.ipynb`  | Safety Alignment & Red-teaming    | AI safety, adversarial testing                | ✅      |
| `nb19_cost_latency_quality.ipynb`      | Cost/Latency/Quality Trade-offs   | Performance optimization, resource management | ✅      |

### 🎯 Part D: Fine-tuning – 6 notebooks

| Notebook                               | Topic                       | Core Skills                                 | Status |
| -------------------------------------- | --------------------------- | ------------------------------------------- | ------ |
| `nb20_lora_peft_tuning.ipynb`          | LoRA (PEFT) Tuning          | Parameter-efficient fine-tuning             | ✅      |
| `nb21_qlora_low_vram.ipynb`            | QLoRA Low VRAM Tuning       | Quantized fine-tuning, memory optimization  | ✅      |
| `nb22_adapters_prefix_tuning.ipynb`    | Adapters/Prefix Tuning      | Adapters, prefix optimization               | ✅      |
| `nb23_dataset_curation_cleaning.ipynb` | Dataset Curation & Cleaning | Data quality, cleaning pipeline             | ✅      |
| `nb24_dpo_vs_rlhf.ipynb`               | DPO vs RLHF                 | Preference learning, reinforcement learning | ✅      |
| `nb25_domain_specific_tuning.ipynb`    | Domain-specific Tuning      | Vertical domain adaptation                  | ✅      |

### 🔍 Part E: RAG × Agents – 5 notebooks

| Notebook                               | Topic                             | Core Skills                             | Status |
| -------------------------------------- | --------------------------------- | --------------------------------------- | ------ |
| `nb26_rag_basic_faiss.ipynb`           | **🔥 RAG Basics (FAISS + PDF QA)** | **Vector retrieval, knowledge Q&A**     | ✅      |
| `nb27_multimodal_rag_clip.ipynb`       | Multimodal RAG (CLIP/BLIP)        | Image-text retrieval, multimodal fusion | ✅      |
| `nb28_retrieval_generation_eval.ipynb` | Retrieval & Generation Evaluation | RAG evaluation, metric design           | ✅      |
| `nb29_multi_agent_collaboration.ipynb` | Multi-agent Collaboration         | Agent orchestration, task allocation    | ✅      |
| `nb30_auto_pipeline_endtoend.ipynb`    | Automated End-to-End Pipeline     | End-to-end automation                   | ✅      |

### 🌐 Part F: WebUI & API – 2 notebooks

| Notebook                           | Topic                           | Core Skills                                 | Status |
| ---------------------------------- | ------------------------------- | ------------------------------------------- | ------ |
| `nb31_gradio_chat_ui.ipynb`        | Gradio Chat UI                  | Web UI, interactive interface               | ✅      |
| `nb32_fastapi_docker_deploy.ipynb` | **FastAPI + Docker Deployment** | **Production deployment, containerization** | ✅      |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the project
git clone https://github.com/your-username/ai-llm-practice.git
cd ai-llm-practice

# Create Conda environment
conda create -n llm-lab python=3.10
conda activate llm-lab

# Install core dependencies
pip install -r requirements.txt

# Setup shared cache
export AI_CACHE_ROOT=~/ai_warehouse
mkdir -p $AI_CACHE_ROOT/{hf,torch}
````

### 2. Hardware Requirements

| Tier            | GPU                     | RAM  | Storage | Recommended Use      |
| --------------- | ----------------------- | ---- | ------- | -------------------- |
| **Minimum**     | None                    | 8GB  | 50GB    | Basics, small models |
| **Recommended** | RTX 3080/4070 (8–12GB)  | 16GB | 100GB   | Most experiments     |
| **Ideal**       | RTX 4080/4090 (16–24GB) | 32GB | 200GB   | LLM fine-tuning      |

---

好的 ✅ 我會將剩下的部分（Learning Paths → Core Tech Stack → Supported Models → Features → Learning Progress → FAQ → Contribution → License → Acknowledgements → Contact）完整翻譯成英文，保持原有的結構與格式。

---

```markdown
### 3. Suggested Learning Paths

#### 🎯 **Fast-track Path** (for learners with prior background)
```

1. nb13\_function\_calling\_tools.ipynb     # Function calling
2. nb26\_rag\_basic\_faiss.ipynb           # RAG basics
3. nb29\_multi\_agent\_collaboration.ipynb # Multi-agent
4. nb32\_fastapi\_docker\_deploy.ipynb     # Deployment

```

#### 📚 **Systematic Path** (for beginners)
```

Part A (Foundations) → Part B (Transformer) → Part C (Applications) →
Part D (Fine-tuning) → Part E (RAG + Agent) → Part F (Deployment)

```

#### 🎨 **Project-oriented Path** (for practitioners)
```

1. Choose your target application scenario
2. Learn the relevant core technique notebooks
3. Integrate multiple techniques to build a full application
4. Deploy and optimize performance

````

---

## 🛠️ Core Tech Stack

### 🔥 Main Frameworks & Libraries

```python
# Core AI frameworks
torch>=2.1.0              # PyTorch deep learning framework
transformers>=4.36.0      # HuggingFace Transformers
accelerate>=0.24.0        # Distributed training acceleration
bitsandbytes>=0.41.0      # Quantization optimization

# LLM application frameworks
langchain>=0.1.0          # LLM application development framework
faiss-cpu>=1.7.0          # Vector similarity search
chromadb>=0.4.0           # Vector database

# Fine-tuning and PEFT
peft>=0.7.0               # Parameter-efficient fine-tuning
trl>=0.7.0                # Transformer reinforcement learning

# Web apps and deployment
fastapi>=0.104.0          # High-performance Web API
gradio>=4.0.0             # Rapid Web UI building
uvicorn>=0.24.0           # ASGI server

# Data processing & evaluation
datasets>=2.14.0          # HuggingFace datasets
evaluate>=0.4.0           # Model evaluation metrics
pandas>=2.0.0             # Data analysis
````

### 🎯 Supported Model Families

| Model Family | Size Range | Primary Use                      | Example                                   |
| ------------ | ---------- | -------------------------------- | ----------------------------------------- |
| **Qwen**     | 7B–72B     | Chinese understanding, reasoning | `Qwen/Qwen2.5-7B-Instruct`                |
| **DeepSeek** | 7B–67B     | Coding, reasoning                | `deepseek-ai/deepseek-r1-distill-qwen-7b` |
| **ChatGLM**  | 6B–130B    | Chinese dialogue                 | `THUDM/chatglm3-6b`                       |
| **Llama**    | 7B–70B     | General-purpose tasks            | `meta-llama/Llama-3.1-8B-Instruct`        |
| **Yi**       | 6B–34B     | Bilingual (Chinese & English)    | `01-ai/Yi-1.5-6B-Chat`                    |

---

## 📊 Project Features & Highlights

### 🌟 **Technical Innovations**

1. **🔄 Unified LLM Adapter Design**

   * Supports `transformers`, `llama.cpp`, and `Ollama` backends
   * One unified codebase for multiple inference engines
   * Automatic low-VRAM optimization

2. **🧠 Chinese-first RAG System**

   * Optimized for Chinese tokenization & retrieval
   * Supports simplified/traditional conversion & multilingual retrieval
   * Integrated BGE embedding models

3. **🤖 Modular Multi-agent Architecture**

   * Roles: Research / Planner / Writer / Reviewer
   * Plug-and-play tools & skill system
   * Automatic decomposition of complex tasks

4. **⚡ Production-level Performance Optimization**

   * 4-bit/8-bit quantization support
   * Dynamic batching
   * Automatic GPU memory management

### 🎯 **Learning Experience Design**

* **🔧 Minimal Viable Examples (MVP)**: Each notebook contains directly runnable examples
* **📈 Progressive Difficulty**: Smooth curve from fundamentals to advanced applications
* **🛡️ Error Prevention**: Default low-VRAM settings, automatic fallback
* **📝 Bilingual Annotation**: English code + explanatory notes

### 🚀 **Practical Orientation**

* **📦 Out-of-the-box**: Shared model cache avoids redundant downloads
* **🐳 Containerized Deployment**: Docker + Kubernetes ready
* **📊 Monitoring Integration**: Prometheus + Grafana observability
* **🔒 Security Considerations**: API authentication, rate-limiting, privacy protection

---

## 📈 Learning Progress Tracking

### ✅ Skills Checklist

#### 🎓 **Fundamentals** (Part A + B)

* [ ] PyTorch tensor operations & autograd
* [ ] Custom neural networks & training loops
* [ ] HuggingFace ecosystem usage
* [ ] Transformer architecture understanding
* [ ] Text generation strategies

#### 🚀 **Application Skills** (Part C)

* [ ] LLM-based text generation
* [ ] Function Calling & tool integration
* [ ] Multi-step reasoning & chains of thought
* [ ] Code generation & debugging
* [ ] Multilingual processing & translation

#### 🎯 **Advanced Skills** (Part D)

* [ ] LoRA/QLoRA parameter-efficient fine-tuning
* [ ] Preference learning (DPO/RLHF)
* [ ] Domain-specific model adaptation
* [ ] Dataset quality control
* [ ] Model evaluation & benchmarking

#### 🔍 **Specialization** (Part E + F)

* [ ] RAG-based knowledge systems
* [ ] Multimodal retrieval & fusion
* [ ] Multi-agent orchestration
* [ ] End-to-end automated pipelines
* [ ] Production API deployment

### 📊 **Suggested Timeline**

| Stage         | Duration | Focus                        | Deliverables                        |
| ------------- | -------- | ---------------------------- | ----------------------------------- |
| **Weeks 1–2** | Part A+B | PyTorch & Transformer basics | Modify existing model architectures |
| **Weeks 3–4** | Part C   | LLM application dev          | Build simple LLM apps               |
| **Weeks 5–6** | Part D   | Fine-tuning & optimization   | Domain-specific fine-tuned model    |
| **Weeks 7–8** | Part E   | RAG & Agent systems          | Integrated QA system                |
| **Week 9**    | Part F   | Deployment & Ops             | Production-ready service            |

---

## 🔧 FAQ & Troubleshooting

### ❓ **Environment Issues**

<details>
<summary><strong>Q: CUDA Out of Memory Error</strong></summary>

```python
# Solution 1: Enable 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Solution 2: Reduce batch size
batch_size = 1  # from default 4 → 1

# Solution 3: Gradient checkpointing
model.gradient_checkpointing_enable()
```

</details>

<details>
<summary><strong>Q: Model download failed or too slow</strong></summary>

```bash
# Option 1: Use HF mirror
export HF_ENDPOINT=https://hf-mirror.com

# Option 2: Set proxy
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# Option 3: Manual download and local path
model = AutoModel.from_pretrained("/path/to/local/model")
```

</details>

<details>
<summary><strong>Q: Windows compatibility issues</strong></summary>

```bash
# Use WSL2 + Docker
wsl --install Ubuntu-22.04
# Setup environment inside WSL2

# Or use Conda isolation
conda create -n llm-lab python=3.10
conda activate llm-lab
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

</details>

### 💡 **Performance Tips**

1. **Memory Optimization**:

   * Use `torch.cuda.empty_cache()`
   * Set `device_map="auto"`
   * Disable `use_cache` during inference

2. **Speed Optimization**:

   * Use `torch.compile()` (PyTorch 2.0+)
   * Enable Flash Attention (when supported)
   * Batch inference instead of single samples

3. **Storage Optimization**:

   * Use GGUF quantized models
   * Share cached models
   * Clean up unused weights

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2024 AI LLM Practice Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
---
