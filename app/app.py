import joblib as joblib
import numpy as np
from flask import Flask, request, jsonify,app,url_for,render_template
import numpy as np
import pandas as pd

app= Flask(__name__)


model = joblib.load('../model/model.joblib')
scaler = joblib.load('../model/scaler.joblib')
waterfront_encoder = joblib.load('../model/waterfront_encoder.joblib')
condition_encoder = joblib.load('../model/condition_encoder.joblib')

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    try:

        data = request.json['data']
        print("Received data:", data)

  
        data['waterfront'] = waterfront_encoder.transform([data['waterfront'].strip().upper()])[0]
        data['condition'] = condition_encoder.transform([data['condition'].strip().title()])[0]

        new_data = [list(data.values())]
        print("Formatted data:", new_data)

        scaled_data = scaler.transform(new_data)
        prediction = model.predict(scaled_data)
        output = float(prediction[0])

        return jsonify({'predicted_price': output})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict',methods=['POST'])
def predict():
    try:
  
        form_data = request.form.to_dict()
        print("Received form data:", form_data)

        form_data['waterfront'] = waterfront_encoder.transform(
            [str(form_data['waterfront']).strip().upper()]
        )[0]

        form_data['condition'] = condition_encoder.transform(
            [str(form_data['condition']).strip().title()]
        )[0]


        final_input = [float(value) for value in form_data.values()]
        final_input = scaler.transform(np.array(final_input).reshape(1, -1))

        prediction = model.predict(final_input)[0]
        return render_template(
                "home.html",
                prediction_text="The predicted price of the house is ${:,.2f}".format(round(prediction, 2))
            )
    
    except Exception as e:
        return render_template("home.html", prediction_text=f"Error: {str(e)}")
    

    
if __name__=='__main__':
    app.run(debug=True)

