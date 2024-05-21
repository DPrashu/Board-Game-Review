import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# load the model
regmodel = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('stand.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    fin_input = scaler.transform(np.array(data).reshape(1,-1))
    output = regmodel.predict(fin_input)
    return render_template('home.html',prediction_text="The average rating of game is {}".format(output[0]))
    
if __name__ == "__main__":
    app.run(debug=True)