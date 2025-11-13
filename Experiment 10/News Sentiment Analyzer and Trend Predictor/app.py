from flask import Flask, request, render_template
import pandas as pd
import string
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk

nltk.download('vader_lexicon')

# Setup
app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

# Load and preprocess dataset
df = pd.read_csv('News_Sentiment_dataset.csv')
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Label sentiment using VADER
def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0.2:
        return 'positive'
    elif score < -0.2:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['full_text'].apply(get_sentiment)
df = df[df['sentiment'] != 'neutral']  # binary classification

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['clean_text'] = df['full_text'].apply(preprocess)

# Train model
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Prediction function
def predict_sentiment(text):
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        input_text = request.form['news_text']
        if input_text.strip() != "":
            sentiment = predict_sentiment(input_text)
            prediction = f"Predicted Sentiment: {sentiment.upper()}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
