# Fake News Detection App

This project is a machine learning-based application to detect fake news articles.  
It uses **Natural Language Processing (NLP)** techniques and a **Logistic Regression** model trained on real and fake news datasets.

## Features
- Preprocesses text using NLTK (tokenization, stopwords removal, lemmatization)
- Converts text into numerical features using TF-IDF
- Trains a Logistic Regression model for classification
- Saves the trained model with `joblib` for later use

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

## Dataset
The project uses two datasets:
- **True.csv** → Real news articles  
- **Fake.csv** → Fake news articles  

Both are combined, cleaned, and labeled for training.

## Usage
Run the notebook or app to train and test the model:
```bash
jupyter notebook app.ipynb
```

If you have a Streamlit app:
```bash
streamlit run app.py
```

## Output
- Trained Logistic Regression model saved as a `.joblib` file.
- Classification results for detecting fake vs. real news.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- nltk
- joblib
- streamlit (optional, if running the web app)

---
