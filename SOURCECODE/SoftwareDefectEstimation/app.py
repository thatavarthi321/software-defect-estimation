import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = joblib.load('models/DT.pkl')

@app.route('/')
def home():
    return render_template('index.html')

def preprocess(features):
    x = features[0]
    x = x.split(',')
    x = np.array(x).reshape(1,-1)
    return x.astype(float)


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [x for x in request.form.values()]
    size = len(int_features)
    print(size)
    if(size == 1):
        final_features = preprocess(int_features)

    else:

        final_features = np.array(int_features,dtype=np.float32).reshape(1,-1)

    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 'Succesful':
        output = "Passed All Checks. Your Code has NO DEFECTS!"
    
    else:
        output = "DEFECTS FOUND! Please Recheck the Module."


    return render_template('index.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
