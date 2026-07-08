# 🎫 AI Support Ticket Router

## 🚀 Overview
An AI-powered web application that automatically classifies support tickets into relevant categories using a fine-tuned BERT model.A fine-tuned BERT classifier that automatically routes support tickets to the correct team — Auth, App, Billing, Database, or Infra — with a live confidence score.

Built end-to-end: data prep → model fine-tuning → inference pipeline → interactive Streamlit app.

This project simulates how real-world support systems can intelligently route tickets to the correct teams, reducing manual effort and improving response time.

---


## ❗ Problem Statement
In most organizations, support tickets are manually reviewed and assigned to different teams (Auth, Billing, Database, etc.).

This leads to:
- ⏳ Delays in issue resolution  
- ❌ Human errors in routing  
- 📉 Inefficient support workflows  

---

## 💡 Solution
This project automates ticket classification using Natural Language Processing (NLP).

👉 Input: Raw support ticket text  
👉 Output: Predicted category with confidence score  

---

## Why This Matters

Manual ticket triage is slow and error-prone — someone has to read every ticket and decide which team owns it. This project automates that step using NLP, so tickets get routed instantly and consistently, the way a real support platform (Zendesk, Freshdesk, Jira Service Management) would do it under the hood.

Input: raw ticket text → Output: predicted category + confidence score + full probability breakdown across all classes.


## ⚙️ Features

- 🎯 5-class ticket classification (auth, app, billing, db, infra) using a fine-tuned bert-base-uncased model

-📊 Confidence score with a low-confidence warning flag, so ambiguous tickets can be escalated for human review instead of   silently misrouted

-📈 Full probability distribution across all categories, not just the top prediction

-🎨 Interactive Streamlit UI with one-click example tickets for fast demos

-⚡ Lightweight inference — runs on CPU, no GPU required

---

## 🧠 How It Works

1.User enters raw ticket text in the UI
2.Text is tokenized with a BertTokenizer (max length 64, truncation + padding)
3.The fine-tuned BertForSequenceClassification model runs a forward pass
4.Logits are converted to probabilities via softmax
5.UI displays the predicted label, confidence %, and per-class probability breakdown


Model details: bert-base-uncased fine-tuned with a fresh classification head, AdamW optimizer (lr 5e-5), 8 training epochs, saved via safetensors for fast, secure loading.

---

## 🛠️ Tech Stack

Model: HuggingFace Transformers (BERT), PyTorch
Interface: Streamlit
Data: Pandas
Serialization: Safetensors

---

## 📸 Demo

### 🔹 Application Screenshot
![App Screenshot](https://github.com/Allure815/ai-support-ticket-router/blob/main/ss.png)

### 🎥 Demo Video
[Watch Demo](https://github.com/Allure815/ai-support-ticket-router/blob/main/Demo-Support%20ticket%20router.mp4)

---

## ▶️ How to Run Locally

bash# Clone
git clone https://github.com/Allure815/ai-support-ticket-router.git
cd ai-support-ticket-router

# Set up environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

# Launch the app
streamlit run app.py


Note: update MODEL_PATH in app.py / predict_ticket.py to point to your local trained_model/ directory (currently hardcoded to a Windows path).



🔭 What's Next


Expand the training dataset for stronger generalization on unseen phrasing
Add a REST API layer (FastAPI) so the router can plug into a real ticketing system
Track per-class precision/recall to catch systematic misroutes
Add active-learning loop: low-confidence predictions get logged for retraining



👤 Author

Heeral — https://github.com/Allure815
