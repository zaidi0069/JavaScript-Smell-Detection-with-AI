# test.py
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score

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

# Example of evaluating the model with validation data
# val_texts = ["const getApplicationVersion = async () => {try {const version = await cloudApi.get('/app/version');return version;} catch (error) {console.error(error);return null;}}"]  # Replace with your validation data
# val_labels = [0]  # Replace with your actual labels

# evaluate_model(val_texts, val_labels)

# Example of predicting a new code snippet
# new_snippet = "const getApplicationVersion = async () => {try {const version = await cloudApi.get('/app/version');return version;} catch (error) {console.error(error);return null;}}"

new_snippet= """async function updateBotSettings(settings) {
    const response = await fetch(`/api/bot/settings/update`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
    });

    const result = await response.json();
    if (result.updated) {
        console.log("Bot settings updated successfully.");
        return { success: true, message: "Settings updated." };
    }
    console.log("Failed to update bot settings.");
    return { success: false, message: "Failed to update settings." };
}


"""
predicted_label = predict_code_snippet(new_snippet)
print(f"Predicted label: {predicted_label}")
