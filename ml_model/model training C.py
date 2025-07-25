import joblib
joblib.dump(model, 'fake_news_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')