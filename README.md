# Fake News Detection Chatbot

This project is a transformer-based chatbot that detects whether a news headline is **real or fake**, fine-tuned on a labeled dataset using **DistilBERT**.

## Live Demo

<!-- [Click here to try the chatbot on Streamlit](https://your-streamlit-link.streamlit.app) -->

## Model

- Base: DistilBERT
- Fine-tuned on: Processed real/fake news dataset
- Frameworks: PyTorch + HuggingFace Transformers

## Folder Structure

- `app.py`: Streamlit chatbot code
- `src/`: Data preprocessing and model training scripts
- `saved_model/`: Trained DistilBERT model
- `data/`: Dataset or preprocessing scripts
- `requirements.txt`: Dependencies

## Run Locally

```bash
# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
