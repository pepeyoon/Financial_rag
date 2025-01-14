import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
import streamlit.components.v1 as components

def load_data():
    df_edu = pd.read_csv('california_colleges.csv')
    df_occ = pd.read_csv('msa_occ_wage_only_3columns.csv')
    
    df_edu['INSTNM'] = df_edu['INSTNM'].str.strip()
    df_edu['CIPDESC'] = df_edu['CIPDESC'].str.strip()
    
    return df_edu, df_occ

def query_llm(prompt, model_name, api_key):
    client = OpenAI(api_key=api_key)
    
    try:
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return chat_completion.choices[0].message.content

# Additional random variable inputs
random_variable_1 = st.text_input("Enter first random variable:")
random_variable_2 = st.text_input("Enter second random variable:")
random_variable_3 = st.text_input("Enter third random variable:")