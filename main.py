import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, flash
import pandas as pd
import traceback
import sys
#import sklearn.preprocessing as preprocessing


# The flask app for serving predictions
app = Flask(__name__)
app.secret_key = 'super secret key'
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
    flash('to_numeric success')
    df['Ward_Type'] = df['Ward_Type'].apply(change)
    df['Severity of Illness'] = df['Severity of Illness'].apply(change1)
    df['Department'] = df['Department'].apply(change3)
    df['Type of Admission'] = df['Type of Admission'].apply(change4)
    flash('convert success')
    prediction1 = kn.predict(df)
    flash('predict success')
    
    final_prediction = key_value.get(prediction1[0])
    flash('return value')
    return render_template('home.html',prediction_text="Estimated staying time {}".format(final_prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    # for direct API calls through request

    json_payload = request.get_json(force=True)
    inference_payload = pd.DataFrame(json_payload, index=[0])
    inference_payload.drop(['Hospital_type_code','City_Code_Hospital','Hospital_region_code','Ward_Facility_Code','Bed Grade','City_Code_Patient','Visitors with Patient','Age'],inplace=True,axis=1)
    inference_payload['Ward_Type']=inference_payload['Ward_Type'].apply(change)
    inference_payload['Severity of Illness'] = inference_payload['Severity of Illness'].apply(change1)
    inference_payload['Department'] = inference_payload['Department'].apply(change3)
    inference_payload['Type of Admission'] = inference_payload['Type of Admission'].apply(change4)

    predict1 = kn.predict(inference_payload)
    #inference_payload['predict'] = predict1
    final_prediction = key_value.get(predict1[0])

    

    #inference_payload['value'] = inference_payload.predict.replace(key_value)
    return render_template('home.html',prediction_text='Estimated staying time {}'.format(final_prediction))
    #return jsonify(inference_payload['value'])

@app.errorhandler(500)
def internal_server_error(e):
    print("500 error occurs")
    etype, value, tb = sys.exc_info()
    traceback.print_exception(etype, value, tb)
    
if __name__ == '__main__':
    
    #app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)

