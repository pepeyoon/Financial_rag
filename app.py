import json
import os
import anthropic
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import plotly.graph_objs as go
import streamlit.components.v1 as components
import statistics

def load_data():
    df_edu = pd.read_csv('california_colleges.csv')
    df_occ = pd.read_csv('msa_occ_wage_only_3columns.csv')
    
    df_edu['INSTNM'] = df_edu['INSTNM'].str.strip()
    df_edu['CIPDESC'] = df_edu['CIPDESC'].str.strip()
    
    return df_edu, df_occ

def query_claude_3_5(prompt, api_key):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None

def format_salary(salary):
    try:
        return "${:,.2f}".format(float(salary))
    except (ValueError, TypeError):
        return "N/A"

def save_output_to_file(output, filename="output.txt"):
    if not output:
        print("No output to save.")
        return
        
    with open(filename, "a") as file:
        file.write(output + "\n")

def validate_json_structure(data):
    required_keys = {
        "revisedExplanation": {
            "1. EDUCATION COSTS",
            "2. FINANCIAL AID", 
            "3. CAREER PROJECTION",
            "4. YEARLY BREAKDOWN"
        },
        "comparison": {
            "mainChanges",
            "financialImpact"
        }
    }
    
    if not all(key in data for key in required_keys):
        return False
    return True

def format_revised_projection(original_data, revised_data):
    if not validate_json_structure(revised_data):
        raise ValueError("Invalid revised projection structure")
        
    try:
        return {
            "data": {
                "years": list(range(1, 16)),
                "netWorth": revised_data["4. YEARLY BREAKDOWN"]["netWorthProgression"],
                "income": revised_data["4. YEARLY BREAKDOWN"]["incomeProgression"],
                "expenses": revised_data["4. YEARLY BREAKDOWN"]["expenseProgression"]
            },
            "summary": {
                "totalNetWorth": revised_data["4. YEARLY BREAKDOWN"]["totalNetWorth"],
                "peakNetWorth": revised_data["4. YEARLY BREAKDOWN"]["peakNetWorth"],
                "averageGrowth": revised_data["4. YEARLY BREAKDOWN"]["averageGrowth"]
            }
        }
    except KeyError as e:
        raise KeyError(f"Missing required key in revised projection: {e}")

def validate_projection_data(original_data, revised_data, transition_year):
    """Validate that projection data is consistent"""
    if len(original_data["data"]["years"]) != len(revised_data["data"]["years"]):
        raise ValueError("Year ranges don't match between projections")
        
    for i in range(transition_year - 1):
        if (original_data["data"]["netWorth"][i] != revised_data["data"]["netWorth"][i] or
            original_data["data"]["income"][i] != revised_data["data"]["income"][i]):
            raise ValueError("Pre-transition data should match original projection")

    return True


def main():
    st.title("15 Year Net Worth Projection")
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if "responses_history" not in st.session_state:
        st.session_state["responses_history"] = []
    if "current_projection" not in st.session_state:
        st.session_state["current_projection"] = None

    df_edu, df_occ = load_data()
    
    # Only Claude 3.5 in this model selection, but keep design
    llm_models = {
        "Claude 3.5": "claude-3.5"
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
    Provide a detailed financial projection with these exact sections:

    1. EDUCATION COSTS
    - Institution: {institution}
    - Program: {field}
    - Residency Status: [determine if {zipcode} is in-state or out-of-state for {institution}]
    - Annual tuition: [exact number]
    - Living costs during school: [exact number]
    - Total 4-year cost: [exact number]

    2. FINANCIAL AID
    - Zipcode {zipcode} median household income: [exact number]
    - Expected grants: [exact number]
    - Loan amount needed: [exact number]
    - Monthly loan payment: [exact number]

    3. CAREER PROJECTION
    - Position: {occupation}
    - Location: {area}
    - Starting salary: [exact number]
    - Expected annual raises: [percentage]
    - Housing ({house} bedroom) monthly cost: [exact number]

    4. YEARLY BREAKDOWN
    Year 1-4 (School):
    - Annual expenses: [exact number]
    - Loan accumulation: [exact number]
    - Net worth change: [exact number]

    Years 5-15 (Career):
    - Annual income: [exact number]
    - Annual expenses: [exact number]
    - Loan payments: [exact number]
    - Savings rate: [exact number]
    - Net worth change: [exact number]

    Provide all numbers as plain numbers without currency symbols or commas.
    Do not include any explanatory text between sections.
    Each number should be a specific value, not a range.
    """

    if st.button("Get Explanation"):
        model_id = llm_models[selected_model]
        explanation_response = query_claude_3_5(explanation_prompt, api_key)
        st.session_state["explanation_response"] = explanation_response
        st.write(explanation_response)
        save_output_to_file(explanation_response)

    if st.session_state["explanation_response"]:
        chart_prompt = f"""
        Generate a financial projection as valid JSON with exactly this structure:
        {{
            "data": {{
                "years": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                "netWorth": [list of 15 numbers],
                "income": [list of 15 numbers],
                "expenses": [list of 15 numbers],
                "loans": [list of 15 numbers]
            }},
            "summary": {{
                "totalNetWorth": number,
                "peakNetWorth": number,
                "averageGrowth": number
            }}
        }}

        Rules:
        - All numbers should be integers or decimals without commas
        - First 4 years should show school expenses and loan accumulation
        - Years 5-15 should show career income and expenses
        - Use the following data for calculations:
        {st.session_state["explanation_response"]}

        Return only the JSON object, no additional text.
        """

        if st.button("Show Chart"):
            chart_response = query_claude_3_5(chart_prompt, api_key)
            st.session_state["chart_response"] = chart_response
            
            try:
                data = json.loads(chart_response)
                years = data["data"]["years"]
                
                fig = go.Figure()
                
                # Net Worth line
                fig.add_trace(go.Scatter(
                    x=years,
                    y=data["data"]["netWorth"],
                    name="Net Worth",
                    mode="lines+markers",
                    line=dict(color="green", width=3)
                ))
                
                # Income and Expenses bars
                fig.add_trace(go.Bar(
                    x=years,
                    y=data["data"]["income"],
                    name="Income",
                    marker_color="blue",
                    opacity=0.6
                ))
                
                fig.add_trace(go.Bar(
                    x=years,
                    y=data["data"]["expenses"],
                    name="Expenses",
                    marker_color="red",
                    opacity=0.6
                ))
                
                fig.update_layout(
                    title="15-Year Financial Projection",
                    xaxis_title="Year",
                    yaxis_title="Amount ($)",
                    barmode="group",
                    template="plotly_white",
                    showlegend=True
                )
                
                st.plotly_chart(fig)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Net Worth", f"${data['summary']['totalNetWorth']:,.0f}")
                with col2:
                    st.metric("Peak Net Worth", f"${data['summary']['peakNetWorth']:,.0f}")
                with col3:
                    st.metric("Average Growth", f"${data['summary']['averageGrowth']:,.0f}")
            
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {str(e)}")
            except KeyError as e:
                st.error(f"Missing required data: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state["chart_response"]:
        plan_prompt = f"""
        Based on the explanation and the JSON chart below, provide 3 alternative plans for the user:
        alternative college, field of study, career, and location recommended. Provide them in concise text.

        Explanation:
        {st.session_state["explanation_response"]}

        JSON Chart:
        {st.session_state["chart_response"]}
        """

        if st.button("Generate Recommended Plans"):
            model_id = llm_models[selected_model]
            plan_response = query_claude_3_5(plan_prompt, api_key)
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
        ("Do you have any hobbies? If so, what kind?", "hobbies"),
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
            # Add current response to history
            st.session_state["responses_history"].append({
                "question": question,
                "response": response
            })
            
            # Build cumulative context
            context = "\n".join([
                f"- {item['question']}: {item['response']}" 
                for item in st.session_state["responses_history"]
            ])
            
            additional_prompt = f"""
            Considering these personal preferences or interests:
            {context}
            
            Generate a financial projection as valid JSON with exactly this structure:
            {{
                "data": {{
                    "years": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                    "netWorth": [list of 15 numbers showing significant changes based on preferences],
                    "income": [list of 15 numbers showing significant changes based on preferences],
                    "expenses": [list of 15 numbers showing significant changes based on preferences],
                    "loans": [list of 15 numbers showing significant changes based on preferences]
                }},
                "summary": {{
                    "totalNetWorth": number,
                    "peakNetWorth": number,
                }},
                "impact": {{
                    "changes": [list of specific changes based on preferences],
                    "financialEffect": [list of financial impacts]
                }}
            }}

            Based on the original projection, ensure that the changes reflected are meaningful and align with the preferences. 
            Additionally, compare these changes to ensure they differ from the previous projections below:
            {st.session_state["explanation_response"]}
            """

            revised_response = query_claude_3_5(additional_prompt, api_key)
            try:
                revised_data = json.loads(revised_response)
                original_data = json.loads(st.session_state["chart_response"])
                years = original_data["data"]["years"]
                
                fig = go.Figure()
                
                # Original projections (dotted lines)
                fig.add_trace(go.Scatter(
                    x=years,
                    y=original_data["data"]["netWorth"],
                    name='Original Net Worth',
                    line=dict(color='blue', dash='dot')
                ))
                
                # Revised projections (solid lines)
                fig.add_trace(go.Scatter(
                    x=years,
                    y=revised_data["data"]["netWorth"],
                    name='Revised Net Worth',
                    line=dict(color='blue')
                ))
                
                # Add other metrics similarly
                for metric, color in [("income", "green"), ("expenses", "red")]:
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=original_data["data"][metric],
                        name=f'Original {metric.title()}',
                        line=dict(color=color, dash='dot')
                    ))
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=revised_data["data"][metric],
                        name=f'Revised {metric.title()}',
                        line=dict(color=color)
                    ))
                
                fig.update_layout(
                    title='15-Year Financial Projection Comparison',
                    xaxis_title='Years',
                    yaxis_title='Amount ($)',
                    showlegend=True
                )
                
                st.plotly_chart(fig)
                
                # Show impact analysis
                st.write("### Impact Analysis")
                for change, effect in zip(
                    revised_data["impact"]["changes"], 
                    revised_data["impact"]["financialEffect"]
                ):
                    st.write(f"- **Change**: {change}")
                    st.write(f"  **Effect**: {effect}")
                    
                # Update current projection
                st.session_state["current_projection"] = revised_data
                
            except json.JSONDecodeError:
                st.error("Failed to parse the projection data")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        if st.button("Change Career/Education Path"):
            st.session_state.show_transition = True

        if "show_transition" in st.session_state and st.session_state.show_transition:
            transition_year = st.slider("Select year to make the change", 1, 15, 5, key="trans_year")
            
            st.subheader(f"New Path Starting Year {transition_year}")
            col1, col2 = st.columns(2)
            
            with col1:
                new_institution = st.selectbox(
                    "New Institution", 
                    df_edu['INSTNM'].unique(), 
                    key="new_inst"
                )
                new_fields = df_edu[df_edu['INSTNM'] == new_institution]['CIPDESC'].unique()
                new_field = st.selectbox(
                    "New Field of Study", 
                    new_fields, 
                    key="new_field"
                )

            with col2:
                new_area = st.selectbox(
                    "New Geographic Area", 
                    df_occ['AREA_TITLE'].unique(), 
                    key="new_area"
                )
                new_occupations = df_occ[df_occ['AREA_TITLE'] == new_area]['OCC_TITLE'].unique()
                new_occupation = st.selectbox(
                    "New Occupation", 
                    new_occupations, 
                    key="new_occ"
                )

            if st.button("Calculate Path Change Impact"):
                transition_prompt = f"""
                Generate a financial projection showing impact of career/education change
                make sure new path shows the negative or positive impact on the net worth and other financial metrics:
                Original Path (Years 1-{transition_year-1}):
                - Institution: {institution}
                - Field: {field}
                - Location: {area}
                - Occupation: {occupation}

                New Path (Years {transition_year}-15):
                - Institution: {new_institution}
                - Field: {new_field}
                - Location: {new_area}
                - Occupation: {new_occupation}

                Return valid JSON with structure:
                {{
                    "data": {{
                        "years": [1-15],
                        "netWorth": [15 numbers],
                        "income": [15 numbers],
                        "expenses": [15 numbers],
                        "loans": [15 numbers]
                    }},
                    "summary": {{
                        "totalNetWorth": number,
                        "peakNetWorth": number,
                        "averageGrowth": number,
                        "transitionCost": number
                    }},
                    "impact": {{
                        "changes": [strings],
                        "financialEffect": [strings]
                    }}
                }}
                """
                
                transition_response = query_claude_3_5(transition_prompt, api_key)
                try:
                    transition_data = json.loads(transition_response)
                    original_data = json.loads(st.session_state["chart_response"])
                    
                    # Filter data to start from the transition year
                    transition_index = transition_year - 1
                    filtered_years = original_data["data"]["years"][transition_index:]
                    filtered_original_net_worth = original_data["data"]["netWorth"][transition_index:]
                    filtered_transition_net_worth = transition_data["data"]["netWorth"][transition_index:]
                    
                    fig = go.Figure()
                    
                    # Original path
                    fig.add_trace(go.Scatter(
                        x=filtered_years,
                        y=filtered_original_net_worth,
                        name="Original Path",
                        mode="lines+markers",
                        line=dict(color="blue", width=2, dash="dash")
                    ))
                    
                    # New path
                    fig.add_trace(go.Scatter(
                        x=filtered_years,
                        y=filtered_transition_net_worth,
                        name="New Path",
                        mode="lines+markers",
                        line=dict(color="green", width=3)
                    ))
                    
                    # Add transition line
                    fig.add_vline(
                        x=transition_year,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Career Change (Year {transition_year})"
                    )
                    
                    fig.update_layout(
                        title="Financial Impact of Career/Education Change",
                        xaxis_title="Year",
                        yaxis_title="Amount ($)",
                        template="plotly_white",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Display transition metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        diff_worth = transition_data["summary"]["totalNetWorth"] - original_data["summary"]["totalNetWorth"]
                        st.metric("Net Worth Impact", f"${diff_worth:,.0f}")
                    with col2:
                        st.metric("Transition Cost", f"${transition_data['summary']['transitionCost']:,.0f}")

                    
                    # Impact analysis
                    st.subheader("Transition Impact Analysis")
                    for change in transition_data["impact"]["changes"]:
                        st.write(f"• {change}")
                        
                except Exception as e:
                    st.error(f"Error calculating transition impact: {str(e)}")

if __name__ == "__main__":
    main()