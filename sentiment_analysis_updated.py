
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load Reddit titles
df = pd.read_csv("reddit_posts.csv")

# Hybrid sentiment scorer (TextBlob + VADER)
sia = SentimentIntensityAnalyzer()

def hybrid_score(text):
    tb = TextBlob(text).sentiment.polarity
    vd = sia.polarity_scores(text)['compound']
    return round((tb + vd) / 2, 3)

def classify_sentiment(score):
    if score > 0.2:
        return "A"  # Strong positive
    elif score < -0.2:
        return "C"  # Negative
    else:
        return "B"  # Neutral

df["sentiment_score"] = df["title"].apply(hybrid_score)
df["rating"] = df["sentiment_score"].apply(classify_sentiment)

# Save
df.to_csv("reddit_sentiments.csv", index=False)
print(f"✅ Saved {len(df)} rows → reddit_sentiments.csv")
