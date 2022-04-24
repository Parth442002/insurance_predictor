import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import seaborn as sns
import joblib


class MachineLearning:
  def __init__(self,model_name,values):
    self.standard_scaler=joblib.load('models/standard_scaler.joblib')
    self.linear_regression=joblib.load('models/linear_regression.joblib')
    self.polynomial_features=joblib.load('models/polynomial_features.joblib')
    self.polynomial_regression=joblib.load('models/polynomial_regression.joblib')
    self.random_forest=joblib.load('models/random_forest.joblib')
    self.svr=joblib.load('models/svr.joblib')
    self.adaboost=joblib.load('models/adaboost.joblib')


    self.model_name=model_name
    self.columns=['age','sex','bmi','children','smoker','region']
    self.values=values

    user_values=dict(zip(self.columns,self.values))

    self.dataset=pd.DataFrame(columns=self.columns)
    self.dataset=self.dataset.append(user_values,ignore_index=True)

  def calculate(self):
    new_variable = {
        "sex": {"male": 0, "female": 1},
        "smoker": {"yes": 0, "no": 1},
        "region": {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3},
      }
    self.dataset.replace(new_variable, inplace=True)
    x_pred=self.dataset.iloc[0,:].values
    x_pred=self.standard_scaler.transform([x_pred])


    if self.model_name=='Linear Regression':
      prediction=self.linear_regression.predict(x_pred)
      return prediction.tolist()[0]

    elif self.model_name=='Polynomial Regression':
      x_pred_poly=self.polynomial_features.transform(x_pred)
      prediction=self.polynomial_regression.predict(x_pred_poly)
      return prediction.tolist()[0]

    elif self.model_name=='Random Forest':
      prediction=self.random_forest.predict(x_pred)
      return prediction.tolist()[0]

    elif self.model_name=='Support Vector Machines':
      prediction=self.svr.predict(x_pred)
      return prediction.tolist()[0]

    elif self.model_name=='Adaboost':
      prediction=self.linear_regression.predict(x_pred)
      return prediction.tolist()[0]
    else:
      return "We have encountered an Error. Please Try again"
