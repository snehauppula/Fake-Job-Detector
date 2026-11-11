# Fake Job Detector

A Streamlit app that flags fraudulent job postings using a machine learning model, with a confidence score and an explanation of the words that drove the prediction.

## How it works

1. Paste a job description into the app
2. A TF-IDF vectorizer transforms the text, and a Logistic Regression model predicts whether the posting is real or fake
3. The app shows the fake-probability score and highlights the words that pushed the prediction, using the model's learned weights

## Files

- `app.py` - the Streamlit app
- `index.ipynb` - training and evaluation notebook
- `fake_job_postings.csv` - the dataset used for training
- `model.pkl` / `vectorizer.pkl` - the saved model and vectorizer the app loads

## Tech stack

Python, scikit-learn (TF-IDF + Logistic Regression), Streamlit, pandas, numpy

## Getting started

```bash
pip install streamlit scikit-learn pandas numpy
streamlit run app.py
```
