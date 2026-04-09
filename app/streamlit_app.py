import streamlit as st
import pandas as pd
import joblib

preproc = None
model = None

st.set_page_config(page_title='Automobile Loan Default Predictor')

st.title('Automobile Loan Default Predictor')

st.sidebar.header('Model & Preprocessor')
preproc_path = st.sidebar.text_input('Preprocessor joblib path', 'models/preprocessor.joblib')
model_path = st.sidebar.text_input('Model joblib path', 'models/loan_model.joblib')

if st.sidebar.button('Load'):
    try:
        preproc = joblib.load(preproc_path)
        model = joblib.load(model_path)
        st.sidebar.success('Loaded.')
    except Exception as e:
        st.sidebar.error(f'Error loading: {e}')
        preproc = None
        model = None

Client_Income = st.number_input('Client Income', min_value=0.0, value=25000.0)
Credit_Amount = st.number_input('Credit Amount', min_value=0.0, value=500000.0)
Loan_Annuity = st.number_input('Loan Annuity', min_value=0.0, value=50000.0)

if st.button('Predict'):
    df_in = pd.DataFrame([{
        'Client_Income': Client_Income,
        'Credit_Amount': Credit_Amount,
        'Loan_Annuity': Loan_Annuity
    }])
    try:
        X = preproc.transform(df_in)
        proba = model.predict_proba(X)[:,1][0]
        st.metric('Default probability', f"{proba:.4f}")
        st.write('Use business threshold to convert probability to decision')
    except Exception as e:
        st.error(f'Inference error: {e}')
