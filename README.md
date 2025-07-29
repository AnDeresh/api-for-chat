# LLM Reincarnation Service

Microservice for fine-tuning LLM model on personal messages using LoRA.

## Installation

1. Clone repository:
```bash
git clone https://github.com/AnDeresh/api-for-chat.git
cd api-for-chat
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Install PyTorch with CUDA support for GPU training
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Hugging Face Access:**
   
   The service uses Meta-Llama-3-8B-Instruct which is a **gated model**.
   
   a) **Request access:**
   - Go to: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
   - Click "Request access to this model"
   - Fill the form with your use case
   - Wait for approval (usually 1-24 hours)
   
   b) **Get access token:**
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token" 
   - Select "Read" permissions
   - Copy the token (starts with `hf_...`)
   
   c) **Login via CLI:**
   ```bash
   huggingface-cli login
   # Paste your token when prompted
   ```

## Usage

```bash
# Start service
cd src
uvicorn main:app --reload

# Swagger docs: http://localhost:8000/docs
```

### API Endpoints

**Upload dialog:**
```bash
curl -X POST "http://localhost:8000/upload-text?speaker=User1" -F "file=@examples/chat.json"
```

**Train model:**
```bash
curl -X POST "http://localhost:8000/train?speaker=User1"
```

## Technical Specs

- **Model:** meta-llama/Meta-Llama-3-8B-Instruct
- **Method:** LoRA (r=8, alpha=16, dropout=0.1)
- **Training:** 1 epoch, batch_size=1

## Requirements

- **GPU:** Recommended (12GB+ VRAM)
- **Python:** 3.10+
- **Storage:** ~20GB

## Project Structure

```
api-for-chat/
├── src/
│   ├── main.py              # FastAPI endpoints
│   ├── parser.py            # Dialog processing
│   └── train.py             # LoRA training logic
├── examples/
│   └── chat.json            # Example dialog
├── data/                    # Filtered messages (auto-created)
├── output_lora/             # Trained models (auto-created)
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```