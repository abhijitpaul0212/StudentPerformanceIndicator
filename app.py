# app.py
import streamlit as st
import pandas as pd
import os

from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import load_object, load_dataframe
from src.logger import logging


data = load_dataframe("artifacts", "data.csv")
data.drop(["math score"], axis=1, inplace=True)

st.write("""
# Student Performance Indicator Prediction App

This app predicts the **Student Performance Indicator**!
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')


def user_input_features(data):
    """
    The user_input_features function takes in the dataframe and returns a new dataframe with the user's inputted features.
    
    :param data: Pass the dataframe to the function
    :return: A dataframe with the user input values
    """

    READING_SCORE = st.sidebar.slider('READING SCORE', int(data['reading score'].min()), int(data['reading score'].max()), int(data['reading score'].mean()))
    WRITING_SCORE = st.sidebar.slider('WRITING SCORE', int(data['writing score'].min()), int(data['writing score'].max()), int(data['writing score'].mean()))
    GENDER = st.sidebar.selectbox('GENDER', options=['male', 'female'])
    RACE_ETHNICITY = st.sidebar.selectbox('RACE/ETHNICITY', options=['group A', 'group B', 'group C', 'group D', 'group E'])
    PARENTAL_EDUCATION = st.sidebar.selectbox('PARENTAL EDUCATION', options=["bachelor's degree", 'some college', "master's degree", "associate's degree", 'high school', 'some high school'])
    LUNCH = st.sidebar.selectbox('LUNCH', options=['standard', 'free/reduced'])
    TEST_PREP_COURSE = st.sidebar.selectbox('TEST PREPARATION COURSE', options=['none', 'completed'])

    data = {'gender': str(GENDER),
            'race/ethnicity': str(RACE_ETHNICITY),
            'parental level of education': str(PARENTAL_EDUCATION),
            'lunch': str(LUNCH),
            'test preparation course': str(TEST_PREP_COURSE),
            'reading score': int(READING_SCORE),
            'writing score': int(WRITING_SCORE)
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features(data)
logging.info("User Input: \n{}".format(df.head()))

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(df)

st.header('Prediction of Student Performance Indicator')
st.write(prediction)
st.write('---')
