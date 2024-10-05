import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the saved model
model = RobertaForSequenceClassification.from_pretrained("./codebert_model")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Function to predict a code snippet
def predict_code_snippet(snippet):
    encoding = tokenizer(snippet, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**encoding)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

# Function to evaluate the model
def evaluate_model(val_texts, val_labels):
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    inputs = {key: torch.tensor(val_encodings[key]) for key in val_encodings}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        preds = np.argmax(outputs.logits.numpy(), axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(val_labels, preds)
    print(f"Validation Accuracy: {accuracy}")
    return accuracy

# Function to test the model on data from an Excel file
def test_model_on_excel(file_path, snippet_column, label_column):
    # Load data from Excel
    df = pd.read_excel(file_path)
    
    # Extract code snippets and labels
    snippets = df[snippet_column].tolist()
    labels = df[label_column].tolist()

    # Evaluate model on the Excel data
    evaluate_model(snippets, labels)

    # Predict a single snippet from Excel file
    for snippet in snippets:
        predicted_label = predict_code_snippet(snippet)
        print(f"Code snippet: {snippet[:50]}... Predicted label: {predicted_label}")

# Example usage: replace 'your_file.xlsx', 'code_snippet_column_name', 'label_column_name' with actual names
test_model_on_excel('./test_data.xlsx', 'code', 'label')
