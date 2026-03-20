import os
os.environ['TRANSFORMERS_CACHE'] = r"D:\cache"

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/tickets.csv")

df.columns = df.columns.str.strip()
df = df.dropna()

# Ensure ticket text is string
df['ticket_text'] = df['ticket_text'].astype(str)

print(df.head())
print(df.shape)

# -----------------------------
# ✅ LABEL MAPPING (FIXED ORDER)
# -----------------------------
labels = ['auth', 'app', 'billing', 'db', 'infra']
label_mapping = {label: idx for idx, label in enumerate(labels)}

df['label_num'] = df['label'].map(label_mapping)

# Check for mapping errors
if df['label_num'].isnull().sum() > 0:
    print("❌ ERROR: Some labels not mapped correctly")
    print(df[df['label_num'].isnull()])
    exit()

print("Label mapping:", label_mapping)

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_inputs = tokenizer(
    df['ticket_text'].tolist(),
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors='pt'
)

# -----------------------------
# Dataset class
# -----------------------------
class TicketDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

dataset = TicketDataset(encoded_inputs, df['label_num'].tolist())

# -----------------------------
# DataLoader
# -----------------------------
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# -----------------------------
# Load BERT model
# -----------------------------
num_labels = len(labels)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_labels
)

# 🔥 Reset classifier (fix bias issue)
model.classifier.reset_parameters()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# -----------------------------
# Training loop
# -----------------------------
model.train()

for epoch in range(8):  # 🔥 increased training
    print(f"\nEpoch {epoch+1}")

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed, last loss: {loss.item()}")

# -----------------------------
# Save model
# -----------------------------
model.save_pretrained(r"D:\support_ticket_router\trained_model")
tokenizer.save_pretrained(r"D:\support_ticket_router\trained_model")

print("✅ Training complete. Model saved successfully!")