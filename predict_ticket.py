from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_PATH = r"D:\support_ticket_router\trained_model"

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# keep simple mapping
labels = ['auth', 'app', 'billing', 'db', 'infra']

def predict_ticket(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    predicted_index = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_index].item()

    return labels[predicted_index], confidence


# test
if __name__ == "__main__":
    text = "Unable to access database"
    label, confidence = predict_ticket(text)

    print("Input:", text)
    print("Prediction:", label)
    print("Confidence:", round(confidence, 2))