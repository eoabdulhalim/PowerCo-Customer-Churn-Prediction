# -*- coding: utf-8 -*-

import joblib
import streamlit as st
import pandas as pd
from model_production import Model_Pipeline
import os



model = joblib.load(r"model.pkl")
sc = joblib.load(r"scaler_ohe.pkl")

st.image('bgg.jpg')

sample_data = 'New Customers Data.csv'  # Path to your CSV file in the same repository
with open(sample_data, 'r') as f:
    csv_data = f.read()

st.download_button(
    label="📥 Download Sample Data",
    data=csv_data,
    file_name='sample_data.csv',
    mime="text/csv"
)

uploaded_file = st.file_uploader("Upload your New Data of New/Existing Customers to Identify those at Risk of Churn.",type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    
    
    features = ['channel_sales','cons_12m','cons_gas_12m','cons_last_month','forecast_cons_12m','forecast_cons_year','forecast_discount_energy',
 'forecast_meter_rent_12m','forecast_price_energy_off_peak','forecast_price_energy_peak','forecast_price_pow_off_peak','has_gas',
 'imp_cons','margin_gross_pow_ele','margin_net_pow_ele','nb_prod_act','net_margin','num_years_antig','origin_up','pow_max',
 'var_year_price_off_peak_var','var_year_price_peak_var','var_year_price_mid_peak_var','var_year_price_off_peak_fix','var_year_price_peak_fix',
 'var_year_price_mid_peak_fix','var_year_price_off_peak','var_year_price_peak','var_year_price_mid_peak','var_6m_price_off_peak_var',
 'var_6m_price_peak_var','var_6m_price_mid_peak_var','var_6m_price_off_peak_fix','var_6m_price_peak_fix','var_6m_price_mid_peak_fix',
 'var_6m_price_off_peak','var_6m_price_peak','var_6m_price_mid_peak']
    
    
    result = Model_Pipeline(new_data, features, model, sc)
    st.write("📈⚙️ Predictions for Uploaded Data:")
    
    csv = result.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name='predictions.csv',
        mime="text/csv"
        )
