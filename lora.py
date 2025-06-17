# Problem: Fine-tuning BERT on a laptop CPU or Colab was challenging due to resource constraints and environment differences.
# Action: Used LoRA with dynamic settings via utils.py's is_colab() to fine-tune on a tiny IMDb subset, optimizing for both environments.
# Result: Enabled efficient feedback classification for PMPMaster, ready for AWS and JavaScript integration.

# Install required libraries (run in Colab or local environment)
# For Colab, run: !pip install -q transformers==4.41.2 peft==0.11.1 datasets==2.20.0 evaluate==0.4.2 torch==2.3.0
# For local, ensure these are installed via pip

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np
import utils 
import random

logger=utils.init(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Environment-specific settings
if utils.is_colab():
    batch_size = 8  # Colab may have more memory
    max_length = 128  # Longer sequences for better accuracy
    lora_rank = 8  # Higher rank for Colab's resources
else:
    batch_size = 4  # Smaller batch for laptop CPU
    max_length = 64  # Shorter sequences for memory efficiency
    lora_rank = 4  # Lower rank for laptop

# Load tiny IMDb dataset (simulating PMPMaster feedback)
full_dataset = load_dataset("imdb")
# Randomly sample 100 for train and 20 for test
train_indices = random.sample(range(len(full_dataset['train'])), 100)
test_indices = random.sample(range(len(full_dataset['test'])), 20)
train_sample = full_dataset['train'].select(train_indices)
test_sample = full_dataset['test'].select(test_indices)
dataset = DatasetDict({
    'train': train_sample,
    'test': test_sample
})

if not utils.is_colab():
    logger.info ("train labels:%s", [x["label"] for x in dataset["train"]])
    logger.info ("test labels:%s", [x["label"] for x in dataset["test"]])

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocess data
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Define LoRA configuration
lora_config = LoraConfig(
    r=lora_rank,  # Dynamic rank based on environment
    lora_alpha=lora_rank * 2,  # Scale alpha with rank
    target_modules=["query", "value"],  # Apply LoRA to attention layers
    lora_dropout=0.05,  # Light dropout for efficiency
    bias="none",  # No bias tuning
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Verify low parameter count

# Define evaluation metric
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./lora_bert_pmpmaster_dynamic",
    
    # copilot - this added to avoid an error
    do_eval=True,

    # copilot suggested to add this, otherwise were getting eval_loss key error due to compute_metrics not being called
    label_names=["labels"], 

    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=5,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    fp16=False,  # Disable mixed precision for CPU
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the LoRA adapter for AWS deployment
model.save_pretrained("lora_bert_pmpmaster_dynamic")
tokenizer.save_pretrained("lora_bert_pmpmaster_dynamic")

# Example inference for JavaScript integration
def predict_feedback(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return "Positive" if probs[0][1] > 0.5 else "Negative"

# Test inference
sample_feedback = "The PMP course practice exams were very helpful."
print(f"Feedback: {sample_feedback} -> {predict_feedback(sample_feedback)}")

# Note: For AWS deployment, upload to S3 and use AWS Lambda (Python runtime) with API Gateway for ReactJS front-end integration.