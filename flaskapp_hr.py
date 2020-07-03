from flask import Flask,request
import numpy as np
import pandas as pd
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in=open('dt.pkl','rb')
classifier=pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "welcome"

@app.route('/predict',methods=["GET"])
def predict_empchurn():
    
    """Let's predict employee churn.
    ---
    parameters:  
      - name: satisfaction_level
        in: query
        type: number
        required: true
      - name: last_evaluation
        in: query
        type: number
        required: true
      - name: number_project
        in: query
        type: number
        required: true
      - name: average_montly_hours
        in: query
        type: number
        required: true
      - name: time_spend_company
        in: query
        type: number
        required: true
      - name: Work_accident
        in: query
        type: number
        required: true
      - name: promotion_last_5years
        in: query
        type: number
        required: true
      - name: salary
        in: query
        type: number
        required: true
      - name: IT
        in: query
        type: number
        required: true
      - name: accounting
        in: query
        type: number
        required: true
      - name: hr
        in: query
        type: number
        required: true
      - name: management
        in: query
        type: number
        required: true
      - name: marketing
        in: query
        type: number
        required: true
      - name: product_mng
        in: query
        type: number
        required: true
      - name: sales
        in: query
        type: number
        required: true
      - name: support
        in: query
        type: number
        required: true
      - name: technical
        in: query
        type: number
        required: true
        
    responses:
        200:
            description: The output values
        
    """
    satisfaction_level=request.args.get('satisfaction_level')
    last_evaluation=request.args.get('last_evaluation')
    number_project=request.args.get('number_project')
    average_montly_hours=request.args.get('average_montly_hours')
    time_spend_company=request.args.get('time_spend_company')
    Work_accident=request.args.get('Work_accident')
    promotion_last_5years=request.args.get('promotion_last_5years')
    salary=request.args.get('salary')
    IT=request.args.get('IT')
    accounting=request.args.get('accounting')
    hr=request.args.get('hr')
    management=request.args.get('management')
    product_mng=request.args.get('product_mng')
    sales=request.args.get('sales')
    support=request.args.get('support')
    technical=request.args.get('technical')
    marketing=request.args.get('marketing')
    
    inputs=np.array([[satisfaction_level, last_evaluation, number_project,
       average_montly_hours, time_spend_company, Work_accident,
       promotion_last_5years, salary, IT, accounting, hr,
       management, marketing, product_mng, sales, support,
       technical]]).astype(np.float64)
    prediction=classifier.predict(inputs)
    
    if prediction==1:
        return "The employee is likley to leave"
    else:
        return "The employee is going to stay"
    
    #return "The prediction value is : "+ str(prediction)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
