# Kimi-Audio Fine-Tuning with FSDP and LoRA - Google Colab

## 🎯 Overview

This notebook provides a **complete, user-friendly solution** for fine-tuning the Kimi-Audio model using:
- **FSDP** (Fully Sharded Data Parallel) for multi-GPU training
- **LoRA** (Low-Rank Adaptation) for memory-efficient training
- **Automatic dataset extraction** from ZIP files
- **Support for multiple dataset formats**

## ✨ Key Features

✅ **Simple Input**: Just provide a ZIP file path  
✅ **Auto-Detection**: Automatically detects dataset format (JSON, JSONL, CSV, Audio)  
✅ **Multi-GPU Support**: FSDP for efficient distributed training  
✅ **Memory Efficient**: LoRA reduces memory requirements by 90%+  
✅ **Production Ready**: Includes checkpointing, logging, and evaluation  
✅ **Flexible**: Works with various dataset structures

## 📦 Dataset Requirements

### Supported Formats

Your ZIP file can contain any of these formats:

#### 1. **JSONL Format** (Recommended)
```jsonl
{"text": "Your training text here", "audio_path": "optional/path/to/audio.wav"}
{"text": "Another training sample", "audio_path": "optional/path/to/audio2.wav"}
```

#### 2. **JSON Format**
```json
[
  {"text": "Training sample 1", "transcript": "Alternative text field"},
  {"text": "Training sample 2", "label": "Another text field name"}
]
```

#### 3. **Audio + Transcript**
```
dataset/
  ├── audio1.wav
  ├── audio1.txt  (transcript)
  ├── audio2.wav
  └── audio2.txt
```

#### 4. **CSV Format**
```csv
text,audio_path
"Training sample 1","path/to/audio1.wav"
"Training sample 2","path/to/audio2.wav"
```

### Field Name Flexibility

The code automatically detects these field names:
- **Text fields**: `text`, `transcript`, `transcription`, `label`, `target`
- **Audio fields**: `audio`, `audio_path`, `file`, `path`

## 🚀 Quick Start

### Step 1: Prepare Your Dataset

1. Organize your data in one of the supported formats
2. Create a ZIP file:
   ```bash
   zip -r my_dataset.zip dataset/
   ```
3. Upload to Google Drive or Colab

### Step 2: Upload to Google Colab

1. Open the notebook in Google Colab
2. Go to **Runtime → Change runtime type**
3. Select **GPU** (T4, V100, or A100)
4. Click **Save**

### Step 3: Mount Google Drive (if needed)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Configure the ZIP Path

In **Cell 3**, change this line:

```python
zip_file_path: str = "/content/drive/MyDrive/your_dataset.zip"
```

To your actual ZIP file path:

```python
zip_file_path: str = "/content/drive/MyDrive/kimi_audio_data.zip"
```

### Step 5: Run All Cells

Click **Runtime → Run all** or run cells sequentially

## ⚙️ Configuration Options

### Model Settings
```python
model_name: str = "moonshot-ai/Kimi-Audio"  # Model to fine-tune
max_length: int = 2048                      # Maximum sequence length
```

### LoRA Settings
```python
use_lora: bool = True          # Enable/disable LoRA
lora_r: int = 16              # LoRA rank (lower = fewer params)
lora_alpha: int = 32          # LoRA scaling factor
lora_dropout: float = 0.05    # Dropout for regularization
```

### FSDP Settings
```python
use_fsdp: bool = True                    # Enable FSDP (multi-GPU)
fsdp_sharding_strategy: str = "FULL_SHARD"  # Sharding strategy
```

### Training Settings
```python
num_train_epochs: int = 3                    # Training epochs
per_device_train_batch_size: int = 2         # Batch size per GPU
gradient_accumulation_steps: int = 4         # Gradient accumulation
learning_rate: float = 2e-4                  # Learning rate
```

## 📊 Expected Training Times

| GPU Type | Batch Size | Dataset Size | Estimated Time/Epoch |
|----------|-----------|--------------|---------------------|
| T4       | 2         | 1,000        | ~30 minutes         |
| V100     | 4         | 1,000        | ~15 minutes         |
| A100     | 8         | 1,000        | ~8 minutes          |

## 💾 Output Structure

After training, you'll get:

```
kimi_audio_finetuned/
├── final_model/              # Complete fine-tuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files
├── lora_adapter/             # LoRA adapter (if enabled)
│   └── adapter files
├── checkpoint-500/           # Training checkpoints
├── checkpoint-1000/
└── logs/                     # TensorBoard logs
    └── events.out.tfevents
```

## 🔄 Loading Your Fine-Tuned Model

### Load Complete Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./kimi_audio_finetuned/final_model")
tokenizer = AutoTokenizer.from_pretrained("./kimi_audio_finetuned/final_model")
```

### Load with LoRA Adapter
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("moonshot-ai/Kimi-Audio")
tokenizer = AutoTokenizer.from_pretrained("moonshot-ai/Kimi-Audio")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./kimi_audio_finetuned/lora_adapter")
```

### Generate Text
```python
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Reduce batch size
```python
per_device_train_batch_size: int = 1
gradient_accumulation_steps: int = 8
```

**Solution 2**: Reduce LoRA rank
```python
lora_r: int = 8
lora_alpha: int = 16
```

**Solution 3**: Reduce max length
```python
max_length: int = 1024
```

### ZIP File Not Found

```python
# Check file path
import os
print(os.path.exists("/content/drive/MyDrive/your_dataset.zip"))

# List directory
!ls -lh /content/drive/MyDrive/
```

### Model Not Loading

```python
# Check available models
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("moonshot-ai/Kimi-Audio", trust_remote_code=True)
```

### Dataset Format Issues

The notebook will show you the detected structure. If it's wrong:

```python
# Check extracted structure
!tree extracted_dataset -L 2

# View sample data
import json
with open('extracted_dataset/data.jsonl', 'r') as f:
    print(json.loads(f.readline()))
```

## 📈 Monitoring Training

### TensorBoard
```python
# In a new cell
%load_ext tensorboard
%tensorboard --logdir ./kimi_audio_finetuned/logs
```

### Training Progress
The notebook shows:
- Current loss
- Learning rate
- Training speed (samples/second)
- Estimated time remaining
- GPU memory usage

## 🎓 Advanced Usage

### Custom Data Preprocessing
```python
# Add after extractor.load_dataset()
def custom_preprocess(item):
    # Your custom logic
    item['text'] = item['text'].lower()
    return item

formatted_data = [custom_preprocess(item) for item in formatted_data]
```

### Multi-GPU Training
```python
# Automatic with FSDP
# Just ensure use_fsdp = True and multiple GPUs are available
print(f"Using {torch.cuda.device_count()} GPUs")
```

### Resume from Checkpoint
```python
# In training arguments
resume_from_checkpoint: str = "./kimi_audio_finetuned/checkpoint-1000"

trainer = Trainer(
    # ... other args
    args=training_args
)
trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```

## 📝 Best Practices

1. **Start Small**: Test with 100 samples first
2. **Monitor GPU Memory**: Use `nvidia-smi` to check usage
3. **Save Frequently**: Set `save_steps=100` for initial runs
4. **Use Gradient Checkpointing**: For very large models
5. **Validate Your Data**: Check a few samples before training

## 🆘 Getting Help

If you encounter issues:

1. Check the error message in the notebook
2. Verify your dataset format matches the examples
3. Ensure your ZIP file is properly structured
4. Check GPU memory with `!nvidia-smi`
5. Try reducing batch size or max length

## 📄 License

This code is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- Moonshot AI for the Kimi-Audio model
- Hugging Face for Transformers and PEFT libraries
- PyTorch team for FSDP implementation

---

**Happy Fine-Tuning! 🚀**