from flask import Flask, request, render_template
import joblib

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

model = joblib.load('churn_pred.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        features = [float(data[feature]) for feature in data.keys()]
        prediction = model.predict([features])[0]
        return str(prediction)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
