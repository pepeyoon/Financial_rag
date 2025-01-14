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

def save_output_to_file(output, filename="output.txt"):
    with open(filename, "a") as file:
        file.write(output + "\n")

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
        save_output_to_file(explanation_response)

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
            save_output_to_file(chart_response)

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
            save_output_to_file(plan_response)

    st.subheader(f"Salary Analysis (using {selected_model})")

    if "additional_step" not in st.session_state:
        st.session_state["additional_step"] = 0

    additional_variables = [
        ("Do you have a pet? If yes, what kind?", "pet"),
        ("What is your favorite subject?", "subject"),
        ("What is your main goal in life?", "goal"),
        ("Do you like money?", "like_money"),
        ("Do you like cars?", "like_cars"),
        ("Do you admire any singer? If yes, who?", "admire_singer")
    ]

    if st.button("Next Step"):
        st.session_state["additional_step"] += 1

    step = st.session_state["additional_step"]
    if step > 0 and step <= len(additional_variables):
        question, var_name = additional_variables[step - 1]
        if var_name in ["like_money", "like_cars"]:
            response = st.checkbox(question)
        else:
            response = st.text_input(question)

        if st.button("Revise Projection"):
            additional_prompt = f"""
            Considering the following personal preference or interest:
            - {question}: {response}
            
            Revise the 15-year net worth projection based on this new information.
            and output in the same formatt as the original explanation.
            also include a comparison between the original and revised explanation.

            Original Explanation:
            {st.session_state["explanation_response"]}
            Format:
            1. Revised Explanation:
                - Key Factors:
                - Factor 1: Description
                - Factor 2: Description
                - ...
            - 15-Year Net Worth Projection:
                - Year 1: $X
                - Year 2: $Y
                 - ...
                - Year 15: $Z

            2. Comparison:
            - Original vs Revised:
                - Year 1: Original $X vs Revised $X'
                - Year 2: Original $Y vs Revised $Y'
                - ...
                - Year 15: Original $Z vs Revised $Z'
            """

            model_id = llm_models[selected_model]
            revised_response = query_llm(additional_prompt, model_id, api_key)
            st.write(revised_response)
            save_output_to_file(revised_response)
            save_output_to_file(additional_prompt)
        
        
if __name__ == "__main__":
    main()
