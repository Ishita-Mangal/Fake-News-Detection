import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ----------- HF Model Path -----------
# Replace this with your actual Hugging Face repo name
HF_MODEL_PATH = "ishita-mangal/fake-news-transformer-model"


# ----------- Load Model + Tokenizer -----------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()


# ----------- Predict Function -----------
def classify_news(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
    
    label_map = {0: "Fake", 1: "True"}
    label = label_map[predicted_class.item()]
    confidence_percent = confidence.item() * 100
    return label, confidence_percent


# ----------- Page Config -----------
st.set_page_config(page_title="Fake News Chatbot", page_icon="🔍", layout="wide")

# ----------- Sidebar Info -----------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2721/2721291.png", width=120)
st.sidebar.title("Fake News Detector")
st.sidebar.markdown("""
Built with:
- DistilBERT
- Streamlit
- Hugging Face Transformers
""")
st.sidebar.markdown("Made by Ishita Mangal")

# ----------- Main App UI -----------
st.title("Fake News Detection Chatbot")
st.markdown("Write or paste a news article snippet, and I'll detect whether it's **Fake** or **True** — with confidence! ")

user_input = st.text_area("Paste your news text here:", height=200)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        label, confidence = classify_news(user_input, tokenizer, model)
        if label == "Fake":
            st.error(f"This news is likely FAKE.\n\nConfidence: `{confidence:.2f}%`.\nBetter double-check your sources.")
        else:
            st.success(f"This news is likely TRUE.\n\nConfidence: `{confidence:.2f}%`")

# ----------- Footer -----------
st.markdown("""
<div style='text-align: center;'>
    Model fine-tuned using DistilBERT on real/fake news. | 
    GitHub: <a href='https://github.com/Ishita-Mangal/Fake-News-Detection' target='_blank'>Repo</a>
</div>
""", unsafe_allow_html=True)
