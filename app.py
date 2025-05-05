import streamlit as st
import pandas as pd
import joblib

model = joblib.load('iris_model.pkl')

st.title('Iris Flower Prediction App')

st.sidebar.header('Group Members')
st.sidebar.markdown('''
- Pol Andrei Bulong
- John Enrique Dela Cruz
- Christian Rosario
- Jayser Marquez
- John Michael Giong-an
- Jessie Garde
''')

st.subheader('Enter the Iris flower measurements:')
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button('Predict'):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = int(round(prediction[0]))

    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.success(f'The predicted species is: {species[prediction]}')