from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# load data
dataset = load_dataset("OpenSound/CapSpeech")

# load model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# special tokens
with open("events.txt", "r") as f:
    events = [line.strip() for line in f]
events = ["<"+event.lower().replace(" ", "_")+">" for event in events]
events.append("<B_start>")
events.append("<B_end>")
events.append("<I_start>")
events.append("<I_end>")
special_tokens_dict = {"additional_special_tokens": events}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_toks} special tokens.")
model.resize_token_embeddings(len(tokenizer))

# data preprocessing
def tokenize_fn(example):
    # You can change the delimiter if needed (e.g., "[SEP]", " | ", or nothing)
    combined = example["text"] + " [SEP] " + example["caption"]
    return tokenizer(combined, padding="max_length", truncation=True, max_length=400)

tokenized_dataset = dataset.map(tokenize_fn)
tokenized_dataset = tokenized_dataset.rename_column("speech_duration", "labels")
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# hyperparameters
training_args = TrainingArguments(
    output_dir="./duration_predictor",
    per_device_train_batch_size=256,
    num_train_epochs=2,
    learning_rate=1e-4,
    warmup_steps=1000,
    save_strategy="steps",
    save_steps=3000,
    evaluation_strategy="epoch",
    logging_dir="./logs_dp",
)

# training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train_PT"],
    eval_dataset=tokenized_dataset["validation_PT"],
)
trainer.train()

# test
preds = trainer.predict(tokenized_dataset["test"])
print("Predictions:", preds.predictions[:10])
print("Ground Truth:", preds.label_ids[:10])
