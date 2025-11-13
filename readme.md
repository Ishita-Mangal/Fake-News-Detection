# Fake News Detection Web App

This is a simple web application built using **Streamlit** to detect whether a tweet is about a real disaster or not, based on NLP and Machine Learning.

## Files

- `app.py` : Streamlit app source code
- `best_model.pkl` : Trained Logistic Regression model
- `tfidf_vectorizer.pkl` : Saved TF-IDF vectorizer
- `requirements.txt` : List of Python dependencies
- `README.md` : Documentation (this file)

## How to run locally

Make sure you have Python 3.x installed.

1. Open terminal / command prompt
2. Navigate to this folder:

```bash
cd web_app
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## How it works

User inputs tweet text in the app.

Text is preprocessed: lowercasing, removing URLs/punctuation/stopwords, stemming.

Text is transformed using the saved TF-IDF vectorizer.

Trained Logistic Regression model predicts:

1 → tweet is about a real disaster

0 → tweet is not about a real disaster

## Note

This app uses a model trained as part of the Fake News Detection on Twitter assignment.

## Author

Ishita Mangal
Final Year Undergraduate at IIT Kanpur
