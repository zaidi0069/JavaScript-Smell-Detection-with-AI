import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Excel file with the correct header settings
file_path = './js_wrong_equality_dataset.xlsx'  # Update this path with the correct file path
df = pd.read_excel(file_path, header=None, names=["code", "label"])

# Check the first few rows to ensure the data is loaded correctly
print(df.head())

# Drop any rows where the label is not an integer
df = df[pd.to_numeric(df['label'], errors='coerce').notnull()]

# Convert labels to integers
df['label'] = df['label'].astype(int)

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Tokenize the data
def tokenize_function(example):
    return tokenizer(example["code"], padding="max_length", truncation=True, max_length=128)

train_texts, val_texts, train_labels, val_labels = train_test_split(df['code'].tolist(), df['label'].tolist(), test_size=0.2)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Create a PyTorch dataset
class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are long integers
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CodeDataset(train_encodings, train_labels)
val_dataset = CodeDataset(val_encodings, val_labels)

# Load the pre-trained CodeBERT model
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Define metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./codebert_model")
