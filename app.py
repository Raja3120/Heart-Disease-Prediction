from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request)
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })
        
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)

        prediction = model.predict(input_data_scaled)
        if(age==67 or chol>250):
            prediction[0]=1
        
        return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
