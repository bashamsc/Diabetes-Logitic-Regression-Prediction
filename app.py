#importing libraries
import os
import numpy as np
import pandas as pd
import flask
from sklearn.externals import joblib
from flask import Flask, render_template, request


# Use joblib to load in the pre-trained model

LR_Model = joblib.load('model/Diabetes_Logistic_Regression_Model.pkl')

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()

@app.route('/',methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
     return flask.render_template('main_diabetes.html')
    
    if flask.request.method == 'POST':
        # Extract the input
       Pregnancies = flask.request.form['Pregnancies']
       Glucose = flask.request.form['Glucose']
       BloodPressure = flask.request.form['BloodPressure']
       SkinThickness = flask.request.form['SkinThickness']
       Insulin = flask.request.form['Insulin']
       BMI = flask.request.form['BMI']
       DiabetesPedigreeFunction = flask.request.form['DiabetesPedigreeFunction']
       Age = flask.request.form['Age']


        # Make DataFrame for model
       input_variables = pd.DataFrame([[flask.request.form['Pregnancies'], flask.request.form['Glucose'], flask.request.form['BloodPressure'], flask.request.form['SkinThickness'],flask.request.form['Insulin'],flask.request.form['BMI'],flask.request.form['DiabetesPedigreeFunction'],flask.request.form['Age']]],
                                       columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
       prediction = LR_Model.predict_proba(input_variables)[::,1] 
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
       result = prediction
       return flask.render_template('result.html',
                                     
                                     result=prediction * 100
                                     
                                     )

if __name__ == "__main__":
 app.run(debug=True)