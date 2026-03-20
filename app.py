import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# -------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0e1117;
}

/* Force base text visible */
html, body, [class*="css"]  {
    color: white !important;
}

/* Headings */
h1, h2, h3 {
    color: white !important;
}

/* Labels */
label {
    color: white !important;
}

/* Buttons */
.stButton>button {
    background-color: #262730;
    color: white !important;
    border-radius: 8px;
}

/* Text area */
textarea {
    background-color: #1c1f26 !important;
    color: white !important;
}

/* Placeholder */
textarea::placeholder {
    color: #bbbbbb !important;
}

/* ✅ FIX RESULT TEXT VISIBILITY */
.stMarkdown, .stText, .stWrite {
    color: #ffffff !important;
    font-weight: 500;
}

/* ✅ Fix probability list */
p {
    color: #e6e6e6 !important;
}

/* ✅ Confidence text */
.stProgress + div {
    color: #ffffff !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)
# -------------------------
# Setup
# -------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

MODEL_PATH = r"D:\support_ticket_router\trained_model"

model = BertForSequenceClassification.from_pretrained(MODEL_PATH, use_safetensors=True)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

labels = ['auth', 'app', 'billing', 'db', 'infra']

# -------------------------
# UI CONFIG
# -------------------------
st.set_page_config(page_title="AI Ticket Router", layout="centered")

st.title("🎫 AI Support Ticket Router")
st.caption("Classifies support tickets into categories using BERT")

# -------------------------
# Session State (Fix examples bug)
# -------------------------
if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""

# -------------------------
# Example Inputs
# -------------------------
st.subheader("Try Examples")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔐 Auth Issue"):
        st.session_state.ticket_text = "User unable to login with OTP"

with col2:
    if st.button("💳 Billing Issue"):
        st.session_state.ticket_text = "Payment deducted but subscription not activated"

with col3:
    if st.button("🗄️ Database Issue"):
        st.session_state.ticket_text = "Database connection timeout error"

# -------------------------
# Input Box
# -------------------------
ticket_text = st.text_area(
    "Enter support ticket text:",
    value=st.session_state.ticket_text
)

# -------------------------
# Prediction
# -------------------------
if st.button("🚀 Predict Category"):
    if ticket_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        inputs = tokenizer(
            ticket_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=64
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        predicted_index = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_index].item()
        predicted_label = labels[predicted_index]

        # -------------------------
        # Output
        # -------------------------
        st.markdown(f"### 🎯 Prediction: **{predicted_label.upper()}**")

        st.progress(confidence)
        st.write(f"Confidence: **{confidence:.2f}**")

        if confidence < 0.5:
            st.warning("⚠️ Low confidence prediction")

        # -------------------------
        # Probabilities
        # -------------------------
        st.subheader("📊 All Category Probabilities")

        for i, label in enumerate(labels):
            st.write(f"{label}: {probs[0][i].item():.2f}")