import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sns.set_style("whitegrid")
sns.set_context("talk")

df = pd.read_csv("X data.csv") 
print(df.head())

sia = SentimentIntensityAnalyzer()

df['score'] = df['clean_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

def vader_sentiment(score):
    if score > 0.05:
        return 1      
    elif score < -0.05:
        return -1    
    else:
        return 0     

df['vader_pred'] = df['score'].apply(vader_sentiment)

accuracy = (df['category'] == df['vader_pred']).mean()
print("Accuracy:", round(accuracy * 100, 2), "%")

plt.figure(figsize=(8,6))

ax = sns.countplot(
    x='vader_pred',
    data=df,
    palette=['#ff4d4d', '#999999', '#4CAF50']
)

for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom'
    )

plt.title("Sentiment Distribution (VADER)", fontsize=16, fontweight='bold')
plt.xlabel("Sentiment (-1=Negative, 0=Neutral, 1=Positive)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,6))

df['vader_pred'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['#ff4d4d', '#999999', '#4CAF50'],
    startangle=90
)

plt.title("Sentiment Share")
plt.ylabel("")
plt.show()
confusion = pd.crosstab(df['category'], df['vader_pred'])

plt.figure(figsize=(6,5))
sns.heatmap(confusion, annot=True, fmt='d', cmap='coolwarm')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['score'], bins=20, kde=True)

plt.title("Sentiment Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()
print(df['category'].value_counts())
print(df['vader_pred'].value_counts())
accuracy = (df['category'] == df['vader_pred']).mean()
print(round(accuracy*100,2))