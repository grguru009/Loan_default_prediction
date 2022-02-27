# Importing libraries
import numpy as np
import pandas as pd
import pickle
import joblib
from flask import Flask, jsonify, request
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


def encode(list_values):
    input_data = pd.DataFrame(
    data=[list_values],
    columns=[  'AGE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'YEARS_EMPLOYED', 'AMT_ANNUITY',
                'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6',
                'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 
                'EXT_SOURCE_2', 
                'AMT_REQ_CREDIT_BUREAU_YEAR',
                'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 
                'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 
                'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_CONTRACT_TYPE', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',  
                'ORGANIZATION_TYPE'     ])
    Num_Inputs=input_data.shape[0]

    
    # Making sure the input data has same columns as it was used for training the model
    # Also, if standardization/normalization was done, then same must be done for new input
  

    #Converting these count columns to float because the following columns have only float/integer values
    ColumnToConvert = ['AGE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'YEARS_EMPLOYED', 'AMT_ANNUITY',
                        'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6',
                        'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 
                        'EXT_SOURCE_2', 
                        'AMT_REQ_CREDIT_BUREAU_YEAR',
                        'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 
                        'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']
    numeric_data = input_data.loc[:,ColumnToConvert]=input_data.loc[:,ColumnToConvert].apply(lambda col: col.astype('float',errors='ignore'))

    cat_columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_CONTRACT_TYPE', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',  
                   'ORGANIZATION_TYPE'     ]

    cat_data = input_data[cat_columns]


    input_data = pd.concat([numeric_data, cat_data], axis=1)

   
    # Appending the new data with the Training data
    DataForML=pd.read_pickle('DataForML.pkl')
    #DataForML=pd.read_pickle(r'C:/AppliedAI/Productionaization/Loan_prediction/DataForML.pkl')

    
    
    InputLoanDetails=input_data.append(DataForML)

    
    # Treating the binary nominal variables first
    InputLoanDetails['CODE_GENDER'].replace({'Male':1, 'Female':0}, inplace=True)
    InputLoanDetails['CODE_GENDER'].replace({'M':1, 'F':0}, inplace=True)
    InputLoanDetails['FLAG_OWN_CAR'].replace({'Y':1, 'N':0}, inplace=True)
    InputLoanDetails['NAME_CONTRACT_TYPE'].replace({'Revolving loans':1, 'Cash loans':0}, inplace=True)
    InputLoanDetails['FLAG_OWN_REALTY'].replace({'Y':1, 'N':0}, inplace=True)
    
    # Generating dummy variables for rest of the nominal variables
    InputLoanDetails=pd.get_dummies(InputLoanDetails)
            
    # Maintaining the same order of columns as it was during the model training
    Predictors=[ 'AGE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'YEARS_EMPLOYED', 'AMT_ANNUITY', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',         'FLAG_PHONE', 
             'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 
             'EXT_SOURCE_2', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 
             'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',  'FLAG_OWN_REALTY',
             
             'NAME_TYPE_SUITE_Unaccompanied', 'NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Spouse, partner', 'NAME_TYPE_SUITE_Children', 
             'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Other_A', 'NAME_TYPE_SUITE_Group of people',
             
             'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_EDUCATION_TYPE_Higher education', 'NAME_EDUCATION_TYPE_Incomplete higher',
             'NAME_EDUCATION_TYPE_Lower secondary', 'NAME_EDUCATION_TYPE_Academic degree',  
               
             'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Single / not married', 'NAME_FAMILY_STATUS_Civil marriage', 
             'NAME_FAMILY_STATUS_Separated', 'NAME_FAMILY_STATUS_Widow', 'NAME_FAMILY_STATUS_Unknown',     

             'NAME_HOUSING_TYPE_House / apartment', 'NAME_HOUSING_TYPE_With parents', 'NAME_HOUSING_TYPE_Municipal apartment',  
             'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_Office apartment', 'NAME_HOUSING_TYPE_Co-op apartment',         

             'ORGANIZATION_TYPE_Business Entity Type 3', 'ORGANIZATION_TYPE_XNA', 'ORGANIZATION_TYPE_Self-employed', 'ORGANIZATION_TYPE_Other',
             'ORGANIZATION_TYPE_Medicine', 'ORGANIZATION_TYPE_Business Entity Type 2', 'ORGANIZATION_TYPE_Government', 'ORGANIZATION_TYPE_School',
             'ORGANIZATION_TYPE_Trade: type 7', 'ORGANIZATION_TYPE_Kindergarten', 'ORGANIZATION_TYPE_Construction', 'ORGANIZATION_TYPE_Business Entity Type 1',
             'ORGANIZATION_TYPE_Transport: type 4', 'ORGANIZATION_TYPE_Trade: type 3', 'ORGANIZATION_TYPE_Industry: type 9', 'ORGANIZATION_TYPE_Industry: type 3',
             'ORGANIZATION_TYPE_Security', 'ORGANIZATION_TYPE_Housing', 'ORGANIZATION_TYPE_Industry: type 11', 'ORGANIZATION_TYPE_Military',
             'ORGANIZATION_TYPE_Bank', 'ORGANIZATION_TYPE_Agriculture', 'ORGANIZATION_TYPE_Police', 'ORGANIZATION_TYPE_Transport: type 2',
             'ORGANIZATION_TYPE_Postal', 'ORGANIZATION_TYPE_Security Ministries', 'ORGANIZATION_TYPE_Trade: type 2', 'ORGANIZATION_TYPE_Restaurant',
             'ORGANIZATION_TYPE_Services', 'ORGANIZATION_TYPE_University', 'ORGANIZATION_TYPE_Industry: type 7', 'ORGANIZATION_TYPE_Transport: type 3',
             'ORGANIZATION_TYPE_Industry: type 1', 'ORGANIZATION_TYPE_Hotel', 'ORGANIZATION_TYPE_Electricity', 'ORGANIZATION_TYPE_Industry: type 4',
             'ORGANIZATION_TYPE_Trade: type 6', 'ORGANIZATION_TYPE_Industry: type 5', 'ORGANIZATION_TYPE_Insurance', 'ORGANIZATION_TYPE_Telecom',
             'ORGANIZATION_TYPE_Emergency', 'ORGANIZATION_TYPE_Industry: type 2', 'ORGANIZATION_TYPE_Advertising', 'ORGANIZATION_TYPE_Realtor',
             'ORGANIZATION_TYPE_Culture', 'ORGANIZATION_TYPE_Industry: type 12', 'ORGANIZATION_TYPE_Trade: type 1',
             'ORGANIZATION_TYPE_Mobile', 'ORGANIZATION_TYPE_Legal Services', 'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Transport: type 1', 
             'ORGANIZATION_TYPE_Industry: type 6', 'ORGANIZATION_TYPE_Industry: type 10', 'ORGANIZATION_TYPE_Religion', 'ORGANIZATION_TYPE_Industry: type 13',
             'ORGANIZATION_TYPE_Trade: type 4', 'ORGANIZATION_TYPE_Trade: type 5', 'ORGANIZATION_TYPE_Industry: type 8',    
                  
             'NAME_INCOME_TYPE_Working', 'NAME_INCOME_TYPE_Commercial associate', 'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State servant', 
             'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Student', 'NAME_INCOME_TYPE_Businessman', 'NAME_INCOME_TYPE_Maternity leave']
    
    # Generating the input values to the model
    X=InputLoanDetails[Predictors].values[0:Num_Inputs]

    
    # Generating the standardized values of X since it was done while model training also
    ScalerFit = joblib.load('PredictorScalerFit.pkl')
    X=ScalerFit.transform(X)
    
    return(X)



@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('model_LR_final.pkl')
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    print('data:')
    print(to_predict_list)

    to_predict_list = encode(to_predict_list)
    print(to_predict_list)
    pred = clf.predict(to_predict_list)
    if pred[0]:
        prediction = "Customer Eligible for Loan"
    else:
        prediction = "Customer Not Eligible for Loan"

    return jsonify({'prediction': prediction})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
