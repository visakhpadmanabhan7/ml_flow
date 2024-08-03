import pickle
import mlflow
from flask import Flask, request, jsonify
mlflow.set_tracking_uri("http://127.0.0.1:8082")
# Set the experiment
mlflow.set_experiment("/model-registry")
app = Flask(__name__)

# Load vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


model_name = "logistic-regression-sentiment-model"
model_version = "latest"

# Load the model from the Model Registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    input_transformed = vectorizer.transform([input_text])
    prediction = model.predict(input_transformed)
    print(prediction)
    return jsonify({'prediction': 'positive' if prediction[0] == 1 else 'negative','score ': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
