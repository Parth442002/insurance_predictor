from turtle import onclick
import streamlit as st
import pandas as pd
import numpy as np
from PipeLine import MachineLearning

st.set_page_config(
    page_icon='🏢',
    page_title='Insurance Cost Predictor',
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This is a insurance cost predictor made using multiple machine learning algorithms such as Random Forests, Support Vector Machines etc",
        'Get Help': 'https://github.com',
        'Report a bug': "https://github.com",
    }
)
st.title("Medical Insurance Predictor")
st.sidebar.title("📊 Choose Your model")
st.sidebar.markdown("""
**Contact Me**:
[Github](https://github.com/Parth442002)
""")

st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

#['age','sex','bmi','children','smoker','region']



st.sidebar.header('🤖 Choose Model')

model_name=st.sidebar.selectbox('Model',
            ('Linear Regression', 'Polynomial Regression', 'Support Vector Machines','Random Forest','Adaboost'))

age=st.sidebar.number_input('Age',value=22,min_value=18,max_value=120,step=1)
sex=st.sidebar.radio('Gender',('Male','Female'))
bmi=st.sidebar.number_input('BMI',value=22.00)
children=st.sidebar.slider('Children',min_value=0,max_value=6)
smoker=st.sidebar.radio('Smoker',('Yes','No'))
region=st.sidebar.selectbox('Region',('NorthEast','NorthWest','SouthEast','SouthWest'))


values=[int(age),sex.lower(),float(bmi),children,smoker.lower(),region.lower()]

print(values)

if st.sidebar.button('Predict'):
  model=MachineLearning(model_name=model_name,
  values=[22, 'male', 22.0, 2, 'yes', 'northeast'])
  prediction=model.calculate()
  prediction=round(prediction,0)
  st.success(f'Your Insurance Cost will be : {prediction}')


@st.cache
def load_data():
  dataset=pd.read_csv('insurance.csv')
  return dataset.head(10)

def model_performance_df():
  model_performance=pd.read_csv('model_performance.csv')
  return model_performance

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data()
model_performace=model_performance_df()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

st.subheader('Model Performance')
st.write(model_performace)

st.subheader('Train Dataset')
st.write(data)






