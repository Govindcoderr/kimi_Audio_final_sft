# Example Dataset Structures

This document shows various valid dataset structures for the Kimi-Audio fine-tuning notebook.

## Structure 1: JSONL with Audio Files

```
my_dataset.zip
└── dataset/
    ├── data.jsonl
    └── audio/
        ├── sample_001.wav
        ├── sample_002.wav
        └── sample_003.wav
```

**data.jsonl content:**
```jsonl
{"text": "Hello, how are you today?", "audio_path": "audio/sample_001.wav"}
{"text": "The weather is nice outside.", "audio_path": "audio/sample_002.wav"}
{"text": "I love machine learning!", "audio_path": "audio/sample_003.wav"}
```

## Structure 2: Simple Text-Only JSONL

```
my_dataset.zip
└── training_data.jsonl
```

**training_data.jsonl content:**
```jsonl
{"text": "This is a training sample for speech synthesis."}
{"text": "Fine-tuning models requires good quality data."}
{"text": "Always validate your dataset before training."}
{"text": "Use diverse examples for better generalization."}
```

## Structure 3: Nested JSON

```
my_dataset.zip
└── data/
    └── conversations.json
```

**conversations.json content:**
```json
[
  {
    "text": "User: What is AI?\nAssistant: AI stands for Artificial Intelligence.",
    "metadata": {"category": "qa"}
  },
  {
    "text": "User: How does machine learning work?\nAssistant: Machine learning uses algorithms to learn patterns from data.",
    "metadata": {"category": "qa"}
  }
]
```

## Structure 4: Audio + Text Files

```
my_dataset.zip
└── recordings/
    ├── audio_001.wav
    ├── audio_001.txt
    ├── audio_002.wav
    ├── audio_002.txt
    ├── audio_003.wav
    └── audio_003.txt
```

**audio_001.txt content:**
```
Hello and welcome to this tutorial on audio processing.
```

**audio_002.txt content:**
```
Today we will learn about speech recognition and synthesis.
```

## Structure 5: CSV Format

```
my_dataset.zip
└── data.csv
```

**data.csv content:**
```csv
text,audio_path,duration
"The quick brown fox jumps over the lazy dog.","audio/fox.wav",3.5
"Machine learning is transforming technology.","audio/ml.wav",4.2
"Always backup your important data.","audio/backup.wav",3.1
```

## Structure 6: Multiple Files

```
my_dataset.zip
├── train/
│   ├── train_001.jsonl
│   └── train_002.jsonl
└── audio/
    └── files/
        ├── aud_001.wav
        └── aud_002.wav
```

**train_001.jsonl:**
```jsonl
{"text": "First training sample", "audio_path": "audio/files/aud_001.wav"}
```

**train_002.jsonl:**
```jsonl
{"text": "Second training sample", "audio_path": "audio/files/aud_002.wav"}
```

## Structure 7: Conversation Format

```
my_dataset.zip
└── conversations.jsonl
```

**conversations.jsonl content:**
```jsonl
{"text": "Human: Tell me about Python.\nAssistant: Python is a high-level programming language known for its simplicity."}
{"text": "Human: What are neural networks?\nAssistant: Neural networks are computing systems inspired by biological neural networks."}
```

## Structure 8: Multi-Modal Dataset

```
my_dataset.zip
└── multimodal/
    ├── metadata.json
    ├── transcripts/
    │   ├── trans_001.txt
    │   └── trans_002.txt
    └── audio/
        ├── rec_001.wav
        └── rec_002.wav
```

**metadata.json:**
```json
[
  {
    "id": "001",
    "text": "Content from trans_001.txt",
    "audio_path": "audio/rec_001.wav",
    "speaker": "person_a"
  },
  {
    "id": "002", 
    "text": "Content from trans_002.txt",
    "audio_path": "audio/rec_002.wav",
    "speaker": "person_b"
  }
]
```

## Structure 9: Instruction-Following Dataset

```
my_dataset.zip
└── instructions.jsonl
```

**instructions.jsonl content:**
```jsonl
{"text": "Instruction: Summarize this text.\nInput: The quick brown fox jumps over the lazy dog.\nOutput: A fox jumps over a dog."}
{"text": "Instruction: Translate to French.\nInput: Hello, how are you?\nOutput: Bonjour, comment allez-vous?"}
```

## Structure 10: Language Pairs

```
my_dataset.zip
└── translations/
    └── en_fr.jsonl
```

**en_fr.jsonl content:**
```jsonl
{"text": "English: Hello | French: Bonjour"}
{"text": "English: Goodbye | French: Au revoir"}
{"text": "English: Thank you | French: Merci"}
```

## Key Points

1. **Flexibility**: The code auto-detects format, so you don't need to worry about exact structure
2. **Text Fields**: Can be named: `text`, `transcript`, `transcription`, `label`, or `target`
3. **Audio Fields**: Can be named: `audio`, `audio_path`, `file`, or `path`
4. **Multiple Files**: The code will find and merge all data files
5. **Nested Structures**: Supports nested directories

## Creating Your ZIP File

### Linux/Mac:
```bash
# From the directory containing your dataset folder
zip -r my_dataset.zip dataset/

# Or compress everything in current directory
zip -r my_dataset.zip .
```

### Windows:
1. Right-click on your dataset folder
2. Select "Send to" → "Compressed (zipped) folder"

### Python:
```python
import zipfile
import os

def create_zip(source_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)

create_zip('dataset/', 'my_dataset.zip')
```

## Minimal Example for Testing

Create this simple structure to test the notebook:

**test_dataset.zip** containing:
```
test.jsonl
```

**test.jsonl content:**
```jsonl
{"text": "This is sample 1"}
{"text": "This is sample 2"}
{"text": "This is sample 3"}
{"text": "This is sample 4"}
{"text": "This is sample 5"}
```

This is the absolute minimum needed to run the training!