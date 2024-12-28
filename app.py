import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

def load_data():
    df_edu = pd.read_csv('college_fos_data.csv')
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

def main():
    st.title("First Year Salary Estimator")
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    df_edu, df_occ = load_data()
    
    llm_models = {
        "GPT-4o": "gpt-4o",
        "GPT-4o-mini": "gpt-4o-mini",
        "GPT-3.5 Turbo": "gpt-3.5-turbo-0125",
        "GPT-o1-mini": "o1-mini",
        "GPT-o1": "o1",
    }
    
    col1, col2, col3 = st.columns([1,2,2])
    
    with col1:
        selected_model = st.selectbox(
            "Select LLM Model",
            list(llm_models.keys())
        )
    
    with col2:
        institution = st.selectbox(
            "Select Institution",
            df_edu['INSTNM'].unique()
        )
        
        fields = df_edu[df_edu['INSTNM'] == institution]['CIPDESC'].unique()
        field = st.selectbox(
            "Select Field of Study",
            fields
        )

    with col3:
        area = st.selectbox(
            "Select Geographic Area",
            df_occ['AREA_TITLE'].unique()
        )
        
        occupations = df_occ[df_occ['AREA_TITLE'] == area]['OCC_TITLE'].unique()
        occupation = st.selectbox(
            "Select Occupation",
            occupations
        )
    
    try:
        salary_info = df_occ[
            (df_occ['AREA_TITLE'] == area) & 
            (df_occ['OCC_TITLE'] == occupation)
        ]['A_MEAN'].iloc[0]
        formatted_salary = format_salary(salary_info)
    except IndexError:
        formatted_salary = "N/A"
    
    print(f"Selected salary: {formatted_salary}")

    if st.button("Generate Salary Estimate"):
        prompt = f"""
        Based on the following information:
        - Institution: {institution}
        - Field of Study: {field}
        - Geographic Area: {area}
        - Occupation: {occupation}
        - Average Salary for Occupation in Area: {formatted_salary}
        
        Please provide a brief estimate of expected first year salary for a graduate, 
        considering the institution's reputation, field of study alignment with occupation,
        and local market conditions. Explain key factors in the estimate.
        Limit the answer to 50 words.
        """
        
        model_id = llm_models[selected_model]
        response = query_llm(prompt, model_id, api_key)
        
        st.subheader(f"Salary Analysis (using {selected_model})")
        st.write(response)
        
if __name__ == "__main__":
    main()