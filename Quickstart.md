# Quick Start Guide - Kimi-Audio Fine-Tuning

## 🎯 Goal
Fine-tune the Kimi-Audio model with your custom dataset using just a ZIP file!

## ⚡ 5-Minute Setup

### Prerequisites
- Google account (for Colab)
- Dataset in a ZIP file
- 15 minutes to 2 hours (depending on dataset size)

## Step-by-Step Guide

### 1. Prepare Your Dataset (5 minutes)

**Option A: Simple Text Data**
Create a file called `data.jsonl`:
```jsonl
{"text": "Your training text here"}
{"text": "Another training sample"}
{"text": "Add as many as you want"}
```

Then create ZIP:
```bash
zip dataset.zip data.jsonl
```

**Option B: Audio + Text**
Create folder structure:
```
dataset/
  ├── audio1.wav
  ├── audio1.txt
  ├── audio2.wav
  └── audio2.txt
```

Then create ZIP:
```bash
zip -r dataset.zip dataset/
```

### 2. Upload to Google Drive (2 minutes)

1. Go to [Google Drive](https://drive.google.com)
2. Upload your `dataset.zip` file
3. Note the path (e.g., `/content/drive/MyDrive/dataset.zip`)

### 3. Open Colab Notebook (1 minute)

1. Open `kimi_audio_sft_colab.ipynb` in Google Colab
2. Or create new notebook and copy-paste the code

### 4. Configure Runtime (1 minute)

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (preferably T4, V100, or A100)
3. Click **Save**

### 5. Configure Your ZIP Path (30 seconds)

In Cell 3, change:
```python
zip_file_path: str = "/content/drive/MyDrive/your_dataset.zip"
```

To:
```python
zip_file_path: str = "/content/drive/MyDrive/dataset.zip"
```

### 6. Mount Google Drive (30 seconds)

Run this in the first cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Click the link and authorize.

### 7. Run All Cells (30 seconds to start)

Click **Runtime** → **Run all**

Or run cells one by one to see progress.

## ⏱️ What Happens Next?

```
[00:00] Installing dependencies... (3-5 min)
[05:00] Extracting your ZIP file... (30 sec)
[05:30] Detecting dataset format... (10 sec)
[05:40] Loading model... (2-5 min)
[10:00] Applying LoRA... (30 sec)
[10:30] Starting training... (varies by dataset)
[END] Saving model and results
```

## 📊 Monitoring Progress

### In Colab
You'll see:
```
Epoch 1/3: 100%|██████████| 100/100 [02:30<00:00,  1.50s/it]
Loss: 0.234 | Learning Rate: 2e-4
```

### TensorBoard
Add this cell and run:
```python
%load_ext tensorboard
%tensorboard --logdir ./kimi_audio_finetuned/logs
```

## 🎉 When Training Completes

You'll see:
```
✅ TRAINING COMPLETED
✅ Model saved to: ./kimi_audio_finetuned/final_model
```

### Download Your Model

```python
# Zip the model
!zip -r finetuned_model.zip kimi_audio_finetuned/

# Download
from google.colab import files
files.download('finetuned_model.zip')
```

### Test Your Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./kimi_audio_finetuned/final_model")
tokenizer = AutoTokenizer.from_pretrained("./kimi_audio_finetuned/final_model")

# Generate
inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## 🔧 Common Adjustments

### Training is Too Slow?
```python
# Reduce epochs
num_train_epochs: int = 1

# Increase batch size (if you have GPU memory)
per_device_train_batch_size: int = 4
```

### Out of Memory?
```python
# Reduce batch size
per_device_train_batch_size: int = 1

# Increase gradient accumulation
gradient_accumulation_steps: int = 8

# Reduce max length
max_length: int = 1024
```

### Want Better Quality?
```python
# More epochs
num_train_epochs: int = 5

# Lower learning rate
learning_rate: float = 1e-4

# Larger LoRA rank
lora_r: int = 32
```

## 📱 Mobile/Tablet Users

1. Use Google Colab mobile app
2. Or use desktop mode in browser
3. All features work the same!

## 💡 Pro Tips

1. **Start with 100 samples** to test everything works
2. **Save checkpoints frequently** in case of interruption
3. **Monitor GPU usage** with `!nvidia-smi` in a new cell
4. **Use descriptive filenames** for your datasets
5. **Keep your ZIP files small** (<2GB for faster upload)

## ⚠️ Important Notes

- Free Colab has **12-hour session limit**
- GPU availability may vary during peak hours
- Large datasets may require Colab Pro
- Always backup your trained models!

## 🆘 Troubleshooting

### "ZIP file not found"
```python
# Check file exists
import os
print(os.path.exists("/content/drive/MyDrive/dataset.zip"))

# List files
!ls -lh /content/drive/MyDrive/
```

### "CUDA out of memory"
```python
# Check GPU
!nvidia-smi

# Reduce batch size in config
per_device_train_batch_size: int = 1
```

### "Model not loading"
```python
# Test model access
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "moonshot-ai/Kimi-Audio",
    trust_remote_code=True
)
```

## 📚 Next Steps

After successful training:
1. Evaluate your model on test data
2. Compare with base model performance
3. Experiment with different hyperparameters
4. Deploy your model to production

## 🔗 Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)

---

**Questions? Issues?** Check the main README.md or DATASET_EXAMPLES.md!

**Happy Training! 🚀**