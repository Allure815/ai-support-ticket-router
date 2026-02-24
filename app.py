import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(
    r"D:\support_ticket_router\trained_model",
    use_safetensors=True  # <-- important for your saved model
)
tokenizer = BertTokenizer.from_pretrained(r"D:\support_ticket_router\trained_model")

# Move model to device
model.to(device)
model.eval()

# Streamlit UI
st.title("Support Ticket Router")

ticket_text = st.text_area("Enter support ticket text:")

if st.button("Predict Category"):
    if ticket_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Tokenize and move to device
        inputs = tokenizer(
            ticket_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=64
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()

        # Label mapping
        label_mapping = {'auth': 0, 'billing': 1, 'db': 2, 'app': 3, 'infra': 4}
        rev_mapping = {v: k for k, v in label_mapping.items()}

        st.success(f"Predicted label: {rev_mapping[predicted_label]}")
