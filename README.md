🚀 Oil & Gas Production ML API

This project provides a Machine Learning-powered REST API that predicts production efficiency based on input parameters. It is designed to integrate seamlessly with RPA tools like UiPath for automated decision-making in industrial workflows.

📌 Overview

The API is built using Python and Flask, trained on production data, and deployed on the cloud. It accepts input data (such as production targets) and returns predicted efficiency values.

This API is part of a larger system that combines:

Automation (UiPath)
Data Processing (Excel)
Machine Learning (Python)
Cloud Deployment

⚙️ Tech Stack
Python
Flask
Scikit-learn
NumPy / Pandas
Cloud Deployment (Render)

🔗 Live API
👉 https://ml-api-project-g7i7.onrender.com

📥 API Usage
Endpoint
POST /predict
Request Body (JSON)
{
  "target": 600
}

Response

{
  "predicted_efficiency": 85.0
}

🛡️ Features
Input validation
Error handling
Anomaly handling (negative and extreme values)
JSON-based communication
Lightweight and fast API

🧠 Model
Model Type: Linear Regression
Input: Target production value
Output: Predicted efficiency

🧪 Testing
The API can be tested using:
Postman
Curl
UiPath HTTP Request Activity
▶️ Run Locally
1. Clone the repository
git clone <https://github.com/teajas14/ml-api-project>
cd <ml_api_project>
2. Install dependencies
pip install -r requirements.txt
3. Run the server
python app.py
4. Access API locally
http://127.0.0.1:5000/predict
📂 Project Structure
.
├── app.py
├── requirements.txt
├── training_data.csv
└── README.md
🚀 Deployment

The API is deployed on Render and can be accessed via the provided URL.

🔄 Integration
This API is designed to integrate with:
UiPath workflows
Excel-based reporting systems
Automation pipelines

📈 Future Improvements
Multi-variable prediction (Oil, Gas, Water)
Advanced ML models
Data preprocessing pipeline
Visualization dashboards

👨‍💻 Author
Teajas Sreejeth

⭐ Acknowledgment
This project demonstrates how automation and machine learning can be integrated to solve real-world industrial problems.
