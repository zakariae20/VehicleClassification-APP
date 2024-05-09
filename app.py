from joblib import load
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = load('C:\\Users\\HP\\Documents\\VehicleClassificationAPP-main\\best_svm_model.joblib')

scaler_subset = StandardScaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['COMPACTNESS']),
            float(request.form['MAX.LENGTH_ASPECT_RATIO']),
            float(request.form['SCALED_VARIANCE_MINOR']),
            float(request.form['MAX.LENGTH_RECTANGULARITY'])
        ]

        features_scaled = scaler_subset.fit_transform([features])

        prediction = model.predict(features_scaled)[0]

        return render_template('result.html', prediction_result=prediction)

    except Exception as e:
        error_message = "An error occurred. Please check the server logs for details."
        return render_template('result.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)