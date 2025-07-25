   from flask import Flask, request, jsonify
   import joblib

   app = Flask(__name__)
   model = joblib.load('fake_news_model.joblib')
   vectorizer = joblib.load('tfidf_vectorizer.joblib')

   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.get_json()
       text = data['text']
       text_vector = vectorizer.transform([text])
       prediction = model.predict(text_vector)[0]
       return jsonify({"prediction": "FAKE" if prediction == 1 else "REAL"})

   if __name__ == '__main__':
       app.run(debug=True)
   