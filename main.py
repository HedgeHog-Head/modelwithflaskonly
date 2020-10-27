import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd
#import sklearn.preprocessing as preprocessing


# The flask app for serving predictions
app = Flask(__name__)
kn = joblib.load(open('kn.pkl','rb'))



@app.route("/")
def home():
    #html = f"<h3>titanic survival prediction home</h3>"
    return render_template('home.html')

def change(ch):
    if(ch=='R'):
        return 0
    elif(ch=='S'):
        return 1
    elif(ch=='Q'):
        return 2
    elif(ch=='P'):
        return 3
    elif(ch=='T'):
        return 4
    elif(ch=='U'):
        return 5

def change1(ch):
    if(ch=='Extreme'):
        return 0
    elif(ch=='Minor'):
        return 1
    elif(ch=='Moderate'):
        return 2

def change3(ch):
    if ch=='radiotherapy':
        return 0
    elif ch== 'anesthesia':
        return 1
    elif ch=='gynecology':
        return 2
    elif ch== 'TB & Chest disease':
        return 3
    elif ch== 'surgery':
        return 4

def change4(ch):
    if ch=='Emergency':
        return 0
    elif ch=='Trauma':
        return 1
    elif ch =='Urgent':
        return 2
key_value={0:'0-10',
        1:'11-20',
        2:'21-30',
        3:'31-40',
        4:'41-50',
        5:'51-60',
        6:'61-70',
        7:'71-80',
        8:'81-90',
        9:'91-100'}

@app.route('/predict',methods = ['POST'])
def predict():
    d = request.form.to_dict()
    df = pd.DataFrame([d.values()],columns=d.keys())
    df.apply(pd.to_numeric,errors='ignore')
    df['Ward_Type'] = df['Ward_Type'].apply(change)
    df['Severity of Illness'] = df['Severity of Illness'].apply(change1)
    df['Department'] = df['Department'].apply(change3)
    df['Type of Admission'] = df['Type of Admission'].apply(change4)
    
    prediction = kn.predict(df)
    
    
    final_prediction = key_value.get(prediction)

    return render_template('home.html',prediction_text="Estimated staying time {}".format(final_prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    # for direct API calls through request

    json_payload = request.get_json(force=True)
    inference_payload = pd.DataFrame(json_payload)
    inference_payload.drop(['Hospital_type_code','City_Code_Hospital','Hospital_region_code','Ward_Facility_Code','Bed Grade','City_Code_Patient','Visitors with Patient','Age'],inplace=True,axis=1)
    inference_payload['Ward_Type']=inference_payload['Ward_Type'].apply(change)
    inference_payload['Severity of Illness'] = inference_payload['Severity of Illness'].apply(change1)
    inference_payload['Department'] = inference_payload['Department'].apply(change3)
    inference_payload['Type of Admission'] = inference_payload['Type of Admission'].apply(change4)

    predict1 = kn.predict(inference_payload)
    inference_payload['predict'] = predict1

    

    inference_payload['value'] = inference_payload.predict.replace(key_value)

    return jsonify(inference_payload['value'])


if __name__ == '__main__':
    app.run(host="127.0.0.1",port=8080,debug=True)

