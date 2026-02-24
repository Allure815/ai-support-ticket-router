import os
os.environ['TRANSFORMERS_CACHE'] = r"D:\cache"


import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load support tickets
df = pd.read_csv("data/tickets.csv", sep="\t")
# Encode labels to numbers
label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label_num'] = df['label'].map(label_mapping)

# Quick check
print(df.head())
print(label_mapping)


# Quick sanity check
print(df.head())
print(df.shape)

from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize all ticket texts
encoded_inputs = tokenizer(
    list(df['ticket_text']),     # your text column
    padding=True,                # pad shorter sentences
    truncation=True,             # truncate longer sentences
    max_length=64,               # max tokens per sentence
    return_tensors='pt'          # return PyTorch tensors
)

# Inspect one example
print(encoded_inputs['input_ids'][0])
print(encoded_inputs['attention_mask'][0])

from torch.utils.data import Dataset, DataLoader

# Create custom dataset
class TicketDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create dataset
dataset = TicketDataset(encoded_inputs, df['label_num'].tolist())

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Inspect one batch
for batch in dataloader:
    print(batch)
    break
from transformers import BertForSequenceClassification
from torch.optim import AdamW


# Number of classes (from your labels)
num_labels = len(label_mapping)

# Load pre-trained BERT with a classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_labels
)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

from torch.nn import CrossEntropyLoss

model.train()  # Set model to training mode

for batch in dataloader:
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss = outputs.loss
    logits = outputs.logits
    
    print(f"Batch loss: {loss.item()}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.save_pretrained(r"D:\support_ticket_router\trained_model")
    tokenizer.save_pretrained(r"D:\support_ticket_router\trained_model")

    print("Model and tokenizer saved successfully!")


