import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify

# Train model
data = {
    "Target": [500, 600, 700, 800, 900],
    "Efficiency": [80, 85, 90, 92, 95]
}

df = pd.DataFrame(data)

X = df[['Target']]
y = df['Efficiency']

model = LinearRegression()
model.fit(X, y)

# Create API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or "target" not in data:
        return jsonify({"error": "Invalid input"}), 400

    target = float(data["target"])

    prediction = model.predict([[target]])[0]

    return jsonify({
        "predicted_efficiency": float(prediction)
    })

# Important for Render
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
