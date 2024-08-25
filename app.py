from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # This will allow all origins

# Load the trained model
model = joblib.load('best_rf.pkl')

# Define the feature names (make sure these match the ones used during training)
feature_names = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30_59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',   
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60_89DaysPastDueNotWorse',
    'NumberOfDependents',
    
]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Convert the features to a DataFrame with column names
        features_df = pd.DataFrame([data['features']], columns=feature_names)
        # Make prediction
        prediction = model.predict(features_df)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
