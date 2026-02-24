from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load trained model and tokenizer
model = BertForSequenceClassification.from_pretrained(r"D:\support_ticket_router\trained_model")
tokenizer = BertTokenizer.from_pretrained(r"D:\support_ticket_router\trained_model")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# Test a new ticket
text = "Unable to access database"
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64).to(device)
outputs = model(**inputs)
predicted_label = torch.argmax(outputs.logits, dim=1).item()
print("Predicted label number:", predicted_label)

# Map number back to text label

# Correct label mapping based on your training data
label_mapping = {'auth': 0, 'app': 1, 'billing': 2, 'db': 3, 'infra': 4}
rev_mapping = {v: k for k, v in label_mapping.items()}

print("Predicted label number:", predicted_label)
print("Predicted label:", rev_mapping[predicted_label])


print("Logits:", outputs.logits)
print("Predicted index:", predicted_label)