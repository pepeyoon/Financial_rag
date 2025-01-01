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
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

def format_salary(salary):
    try:
        return "${:,.2f}".format(float(salary))
    except (ValueError, TypeError):
        return "N/A"

def extract_html(response):
    html_pattern = re.compile(r'<html>.*?</html>', re.DOTALL)
    match = html_pattern.search(response)
    if match:
        return match.group(0)
    return "No HTML content found."

def main():
    st.title("15 Year Net Worth Projection")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    df_edu, df_occ = load_data()
    
    llm_models = {
        "GPT-4o": "gpt-4o",
        "GPT-4o-mini": "gpt-4o-mini",
        "GPT-3.5 Turbo": "gpt-3.5-turbo-0125",
    }
    
    col1, col2, col3 = st.columns([1,2,2])
    
    with col1:
        selected_model = st.selectbox("Select LLM Model", list(llm_models.keys()))
    
    with col2:
        institution = st.selectbox("Select Institution", df_edu['INSTNM'].unique())
        fields = df_edu[df_edu['INSTNM'] == institution]['CIPDESC'].unique()
        field = st.selectbox("Select Field of Study", fields)

    with col3:
        area = st.selectbox("Select Geographic Area", df_occ['AREA_TITLE'].unique())
        occupations = df_occ[df_occ['AREA_TITLE'] == area]['OCC_TITLE'].unique()
        occupation = st.selectbox("Select Occupation", occupations)
    
    zipcode = st.text_input("Enter your current ZIP code")
    
    house = st.number_input('Do you plan to live alone? How many bedrooms?', min_value=1, step=1)
    
    try:
        salary_info = df_occ[
            (df_occ['AREA_TITLE'] == area) & 
            (df_occ['OCC_TITLE'] == occupation)
        ]['A_MEAN'].iloc[0]
        formatted_salary = format_salary(salary_info)
    except IndexError:
        formatted_salary = "N/A"
    

    if "explanation_response" not in st.session_state:
        st.session_state["explanation_response"] = ""

    if "chart_response" not in st.session_state:
        st.session_state["chart_response"] = ""

    explanation_prompt = f"""
    
    Imagine you are a student planning to attend {institution} to study {field}.
    Use {zipcode} zipcode which is the student's current zipcode to get median household income to provide grants, net price of the college also consider if student is resident or non resident.
    Project 15-year income/expenses based on {occupation} which is the career student plans to pursue after graduation, {area} where student plans to work and number of bedroom: {house} where student plans to live after graduation.
    First 4 years as school years (debt only). Explain simple key factors in text.
    """

    if st.button("Get Explanation"):
        model_id = llm_models[selected_model]
        explanation_response = query_llm(explanation_prompt, model_id, api_key)
        st.session_state["explanation_response"] = explanation_response
        st.write(explanation_response)

    if st.session_state["explanation_response"]:
        chart_prompt = f"""
        CHART_ONLY: Provide only the HTML for a 15-year projection graph including income, 
        expenses, tuition, and student loan and also a chart including details based on the explanation below.
        tuition only applies to the first 4 years.
        No additional text and be as concise as possible using the explanation response. no values should be the same since they are projections so consider inflation and change of lifestyle during certain age.

        Explanation:
        {st.session_state["explanation_response"]}

        Also show the total net worth next to the chart.
        """

        if st.button("Show Chart"):
            model_id = llm_models[selected_model]
            chart_response = query_llm(chart_prompt, model_id, api_key)
            st.session_state["chart_response"] = chart_response
            components.html(chart_response, height=400, scrolling=True)

    if st.session_state["chart_response"]:
        plan_prompt = f"""
        RECOMMENDED_PLANS: Based on the explanation and chart below, provide 3 alternative plans for the user, which alternative college, field of study, career, and location you would recommend to the user.
        
        Explanation:
        {st.session_state["explanation_response"]}
        
        Chart:
        {st.session_state["chart_response"]}
        """

        if st.button("Generate Recommended Plans"):
            model_id = llm_models[selected_model]
            plan_response = query_llm(plan_prompt, model_id, api_key)
            st.write(plan_response)

    st.subheader(f"Salary Analysis (using {selected_model})")
        
if __name__ == "__main__":
    main()
