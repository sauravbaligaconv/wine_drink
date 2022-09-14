import pickle
import jsonify
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split




data=pd.read_csv('wine.csv')

X=data.drop(['quality'],axis=1)
y=data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

ada_model = AdaBoostClassifier(n_estimators = 50, random_state = 10,learning_rate= 0.2)
ada_model.fit(X_train, y_train)


filename = 'finalized_model.pkl'
pickle.dump(ada_model, open(filename, 'wb'))


app = Flask(__name__)
model = pickle.load(open('finalized_model.pkl', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        fixed_acidity=float(request.form['fixed_acidity'])
        volatile_acidity=float(request.form['volatile_acidity'])
        citric_acid=float(request.form['citric_acid'])
        residual_sugar=float(request.form['residual_sugar'])
        chlorides=float(request.form['chlorides'])
        total_sulfur_dioxide=float(request.form['total_sulfur_dioxide'])
        density=float(request.form['density'])
        pH=float(request.form['pH'])
        sulphates=float(request.form['sulphates'])
        alcohol=float(request.form['alcohol'])
        sulfur_dioxide_ratio=float(request.form['sulfur_dioxide_ratio'])
        type1=request.form['type1']
        if(type1=='Red'):             
            type1=1 	
        elif(type1=='White'):
            type1=0

        def lr(type1,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,total_sulfur_dioxide,density,pH,sulphates,alcohol,sulfur_dioxide_ratio):
            c=pd.DataFrame([type1,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,total_sulfur_dioxide,density,pH,sulphates,alcohol,sulfur_dioxide_ratio]).T
            return model.predict(c)
          
    
    prediction=lr(type1,fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,total_sulfur_dioxide,density,pH,sulphates,alcohol,sulfur_dioxide_ratio)
    return render_template('index.html',prediction_text="Wine Quality is {}".format(prediction))
  

if __name__=="__main__":
    app.run(debug=True)

