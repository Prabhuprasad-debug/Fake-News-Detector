import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("news_dataset.csv")

# ✅ Basic Cleaning
df["text"] = df["text"].str.lower()  # Convert to lowercase
df["text"] = df["text"].str.replace(r'[^\w\s]+', '')  # Remove special chars

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Convert text to features (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)