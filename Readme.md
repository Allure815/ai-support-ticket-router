# 🎫 AI Support Ticket Router

## 🚀 Overview
An AI-powered web application that automatically classifies support tickets into relevant categories using a fine-tuned BERT model.

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

## ⚙️ Features

- 🎯 Accurate ticket classification (Auth, App, Billing, DB, Infra)
- 📊 Confidence score for predictions
- 📈 Probability distribution across all categories
- 🎨 Clean and interactive UI using Streamlit
- ⚡ Example-based quick testing

---

## 🧠 How It Works

1. Ticket text is input through the UI  
2. Text is tokenized using a BERT tokenizer  
3. A fine-tuned BERT model predicts the category  
4. Output includes:
   - Predicted label  
   - Confidence score  
   - All category probabilities  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- HuggingFace Transformers (BERT)  
- PyTorch  
- Pandas  

---

## 📸 Demo

### 🔹 Application Screenshot
![App Screenshot](https://github.com/Allure815/ai-support-ticket-router/blob/main/ss.png)

### 🎥 Demo Video
[Watch Demo](https://github.com/Allure815/ai-support-ticket-router/blob/main/Demo-Support%20ticket%20router.mp4)

---

## ▶️ How to Run Locally

```bash
# Clone repo
git clone <your-repo-link>

# Navigate
cd support-ticket-router

# Activate virtual environment
venv\Scripts\activate

# Run app
streamlit run app.py
