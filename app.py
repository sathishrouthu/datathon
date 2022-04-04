import numpy as np
from flask import Flask, request, jsonify, render_template,redirect
import pickle
import pandas as pd
#data_cleaned = pd.read_csv('models/data_final.csv')
app = Flask(__name__)
logreg = pickle.load(open('models/logreg.pkl', 'rb'))
dtmodel = pickle.load(open('models/dt.pkl', 'rb'))
rfe = pickle.load(open('models/rfe.pkl', 'rb'))
ensemble = pickle.load(open('models/ensemble.pkl', 'rb'))
svm = pickle.load(open('models/SVM1.pkl', 'rb'))
scaler = pickle.load(open('models/standard_scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST','GET'])
def predictor():
    if request.method == 'POST':
        column_names = ['GENDER', 'AGE', 'WBC', 'Platelets', 'Neutrophils', 'Lymphocytes','Monocytes', 'Eosinophils', 'Basophils', 'CRP', 'AST', 'ALT', 'LDH']
        form_values = request.form.to_dict()  
        print("sathish")  
        d = {col:np.float64(form_values[col]) for col in column_names }
        df = pd.DataFrame(d,index=[0])
        df_s = scaler.transform(df)
        
        df_s=pd.DataFrame(df_s,columns=d.keys())
        df_s['GENDER']=df['GENDER']
        if request.form['MODEL']=='logreg':
            result = logreg.predict(df_s)
        if request.form['MODEL']=='dtree':
            result = dtmodel.predict(df_s)
        if request.form['MODEL']=='rfe':
            result = rfe.predict(df_s)
        if request.form['MODEL']=='svm':
            result = svm.predict(df_s)
        if request.form['MODEL']=='ensemble':
            result = ensemble.predict(df_s)
        print(result)
        #ipdata= pd.DataFrame(input_data,columns=data_cleaned.columns)
        #input_data = scaler.transform(input_data)
        #prediction=dt.predict(input_data)
        #print(prediction)
        return render_template('index.html',result=result[0])
    return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
    