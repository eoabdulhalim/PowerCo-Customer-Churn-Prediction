# -*- coding: utf-8 -*-
"""Model_Production.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11LypuJVuFkiD6beCL4nGuB44cFzI6vCX
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def feature_engineering(data):
  data['diff_year_price_1'] = data['var_year_price_off_peak_var'] - data['var_year_price_mid_peak_var']
  data['diff_year_price_2'] = data['var_year_price_off_peak_var'] - data['var_year_price_peak_var']
  data['diff_year_price_3'] = data['var_year_price_off_peak_fix'] - data['var_year_price_mid_peak_fix']
  data['diff_year_price_4'] = data['var_year_price_off_peak'] - data['var_year_price_mid_peak']
  return data

def EncodingCategoricalFeatures(data,sc, test=False):
  cate = ['channel_sales','origin_up']
  data['has_gas'] = data['has_gas'].map({'t':1,'f':0})
  if test:
    encoded_arr = sc.transform(data[cate])
  else:
    sc = OneHotEncoder(sparse_output=False)
    encoded_arr = sc.fit_transform(data[cate])
  encoded_df = pd.DataFrame(encoded_arr, columns=sc.get_feature_names_out(cate))
  result = pd.concat([data.drop(columns=cate), encoded_df], axis=1)
  return result, sc

def Model_Pipeline(new_data,features, model, sc):
  ids = new_data["id"].reset_index(drop=True)
  channel_sales = new_data["channel_sales"].reset_index(drop=True)
  new_data = new_data[features]
  new_data = feature_engineering(new_data)
  new_data, _ = EncodingCategoricalFeatures(new_data, sc, test=True)
  new_data = np.array(new_data)
  #new_data = scaler.transform(new_data)
  predictions = model.predict(new_data)
  predictions_series = pd.Series(predictions, name="Churn Prediction").reset_index(drop=True)
  predictions_df = pd.DataFrame({
        "ID": ids,
        "Channel Sales": channel_sales,
        "Churn Prediction": predictions_series
    })
  return predictions_df