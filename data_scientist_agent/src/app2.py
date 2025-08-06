import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import re
import io
import os
import time
import contextlib
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --- Configuration & Initialization ---
load_dotenv()

# --- Core Prompts for the Custom Agent ---
CATEGORIZER_PROMPT = """
You are a text classification agent. Your task is to classify the user's request into ONE of the following categories based on their intent.
The categories are: ["Plotting", "Dataframe Info/Summary", "Data Manipulation", "General Question", "Other"].
Respond with ONLY the category name. Do not add any explanation or punctuation.

User Request: "{prompt}"
Category:
"""

AGENT_PROMPT = """
You are Gemini, a friendly and expert data science assistant. Your goal is to help the user analyze their pandas DataFrame (`df`).

**Your Personality:**
- **Helpful & Explanatory:** You don't just write code. You explain your thought process, what the code does, and how to interpret the results.
- **Proactive:** You anticipate user needs and suggest next steps.
- **Structured:** You use markdown (like headers, bolding, and lists) to make your responses easy to read.

**How to Respond:**
1.  **Greeting & Plan:** Start with a friendly greeting. Briefly state your understanding of the user's request and your plan to address it.
2.  **Explanation (If generating code):** Describe the analysis you're about to perform.
3.  **Code Block (If generating code):** Provide the Python code to perform the analysis within a single ```python...``` block.
4.  **Interpretation & Next Steps:** Briefly explain what the user should look for in the output and suggest a logical next question or analysis.
5.  **If no code is needed:** Simply provide a helpful, conversational answer in markdown.

**Important Rules for the Code:**
- The user's data is in a pandas DataFrame called `df`.
- **For plotting, always create a figure and axes object (e.g., `fig, ax = plt.subplots()`) and perform your plotting on the `ax` object. The `fig` object will be displayed automatically.**
- Do not use any Streamlit functions (`st.*`).
- Your code must be self-contained within the markdown block.

Here is the chat history for context:
{history}

**User's Request:** {prompt}
"""

# --- Helper Functions ---
def parse_response(response: str):
    """Parses the LLM response to separate the explanation from the code block."""
    code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        explanation = response[: code_match.start()].strip()
        return explanation, code
    else:
        return response.strip(), None

def execute_code(code: str, df: pd.DataFrame):
    """Executes the provided Python code in a controlled environment."""
    plt.close("all")
    local_vars = {"df": df.copy(), "pd": pd, "plt": plt}
    string_io = io.StringIO()
    try:
        with contextlib.redirect_stdout(string_io):
            exec(code, {"__builtins__": __builtins__}, local_vars)
        
        new_df = local_vars.get("df", df)
        output = None
        
        # Check for matplotlib figures
        fig_vars = [v for v in local_vars.values() if isinstance(v, plt.Figure)]
        if fig_vars:
            output = fig_vars[0]
        elif plt.get_fignums():
            output = plt.gcf()
        else:
            # Check for other outputs
            output = string_io.getvalue()
            if not output:
                try: # Attempt to eval the last line if it's an expression
                    last_line = code.strip().split('\n')[-1]
                    if "=" not in last_line:
                        output = eval(last_line, {"__builtins__": __builtins__}, local_vars)
                except Exception:
                    output = "Code executed successfully with no direct output."
        
        plt.close("all") # Clean up plots
        return output, new_df
    except Exception as e:
        plt.close("all")
        return f"Error executing code: {e}", df

@st.cache_data
def to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode("utf-8")

# --- Page & State Configuration ---
st.set_page_config(page_title="AI Insight Engine", layout="wide")
st.title("🤖 AI Insight Engine")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "usage_stats" not in st.session_state:
    st.session_state.usage_stats = {
        "total_queries": 0, "successful_executions": 0, "failed_executions": 0,
        "query_types": {}, "total_llm_time": 0.0, "total_exec_time": 0.0
    }

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input(
        "Groq API Key", type="password", help="Get your key from https://console.groq.com/keys",
        value=os.environ.get("GROQ_API_KEY", "")
    )
    model_name = st.selectbox("Select Model", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"], index=0)
    
    st.divider()
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        if "uploaded_file_state" not in st.session_state or st.session_state.uploaded_file_state != uploaded_file:
            st.session_state.uploaded_file_state = uploaded_file
            st.session_state.current_df = pd.read_csv(uploaded_file)
            st.session_state.messages = [{"role": "assistant", "content": "Hello! Your data is loaded. The dashboard is ready and you can chat with me below."}]
            # Reset stats for new file
            st.session_state.usage_stats = {
                "total_queries": 0, "successful_executions": 0, "failed_executions": 0,
                "query_types": {}, "total_llm_time": 0.0, "total_exec_time": 0.0
            }
            st.rerun()

    # --- Data Cleaning Tools (from Script 1) ---
    if st.session_state.current_df is not None:
        st.divider()
        st.header("2. Clean & Preprocess")

        with st.expander("Handle Missing Values"):
            df = st.session_state.current_df
            missing_vals_cols = df.columns[df.isnull().any()].tolist()
            if not missing_vals_cols:
                st.info("No columns with missing values.")
            else:
                col_to_clean = st.selectbox("Select column to clean:", options=missing_vals_cols)
                impute_method = st.radio("Choose method:", ("Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows"), horizontal=True)
                if st.button(f"Apply to '{col_to_clean}'"):
                    cleaned_df = df.copy()
                    if impute_method == "Fill with Mean":
                        if pd.api.types.is_numeric_dtype(cleaned_df[col_to_clean]):
                            cleaned_df[col_to_clean].fillna(cleaned_df[col_to_clean].mean(), inplace=True)
                        else: st.warning("Mean only works for numeric columns.")
                    elif impute_method == "Fill with Median":
                        if pd.api.types.is_numeric_dtype(cleaned_df[col_to_clean]):
                            cleaned_df[col_to_clean].fillna(cleaned_df[col_to_clean].median(), inplace=True)
                        else: st.warning("Median only works for numeric columns.")
                    elif impute_method == "Fill with Mode":
                        cleaned_df[col_to_clean].fillna(cleaned_df[col_to_clean].mode()[0], inplace=True)
                    elif impute_method == "Drop Rows":
                        cleaned_df.dropna(subset=[col_to_clean], inplace=True)
                    
                    st.session_state.current_df = cleaned_df
                    st.success(f"Applied '{impute_method}' to '{col_to_clean}'.")
                    st.rerun()

        with st.expander("Change Data Types"):
            df = st.session_state.current_df
            col_to_convert = st.selectbox("Select column to convert:", options=df.columns)
            current_type = str(df[col_to_convert].dtype)
            convert_type = st.selectbox(f"Convert '{col_to_convert}' (currently {current_type}) to:", ("Numeric", "Text (Object)", "Datetime"))
            if st.button(f"Convert '{col_to_convert}'"):
                converted_df = df.copy()
                try:
                    if convert_type == "Numeric":
                        converted_df[col_to_convert] = pd.to_numeric(converted_df[col_to_convert], errors='coerce')
                    elif convert_type == "Text (Object)":
                        converted_df[col_to_convert] = converted_df[col_to_convert].astype(str)
                    elif convert_type == "Datetime":
                        converted_df[col_to_convert] = pd.to_datetime(converted_df[col_to_convert], errors='coerce')
                    st.session_state.current_df = converted_df
                    st.success(f"Converted '{col_to_convert}' to {convert_type}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Conversion failed: {e}")

        st.divider()
        st.header("3. Manage Session")
        if st.button("Reset DataFrame & Chat"):
            if "uploaded_file_state" in st.session_state and st.session_state.uploaded_file_state:
                st.session_state.uploaded_file_state.seek(0)
                st.session_state.current_df = pd.read_csv(st.session_state.uploaded_file_state)
                st.session_state.messages = [{"role": "assistant", "content": "DataFrame and chat have been reset."}]
                st.success("Session has been reset to the original data.")
                st.rerun()

        csv_data = to_csv(st.session_state.current_df)
        st.download_button(label="Download Modified CSV", data=csv_data, file_name="cleaned_data.csv", mime="text/csv")

    # --- Usage Statistics Panel (from Script 2) ---
    st.divider()
    st.header("📊 Usage Statistics")
    stats = st.session_state.usage_stats
    col1, col2 = st.columns(2)
    col1.metric("Total Queries", stats['total_queries'])
    col2.metric("✅ Success", stats['successful_executions'])
    col2.metric("❌ Failed", stats['failed_executions'])
    avg_llm_time = (stats['total_llm_time'] / stats['total_queries']) if stats['total_queries'] > 0 else 0
    avg_exec_time = (stats['total_exec_time'] / stats['successful_executions']) if stats['successful_executions'] > 0 else 0
    st.metric("Avg LLM Time", f"{avg_llm_time:.2f}s")
    st.metric("Avg Exec Time", f"{avg_exec_time:.2f}s")
    st.write("Query Type Distribution:")
    if stats['query_types']:
        st.bar_chart(pd.DataFrame.from_dict(stats['query_types'], orient='index', columns=['count']))

# --- Main Interface ---
if st.session_state.current_df is None:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to get started.")
else:
    df = st.session_state.current_df
    dashboard_tab, chat_tab = st.tabs(["📊 Automated Dashboard", "💬 Chat with Data"])

    # --- Automated Dashboard Tab (from Script 1) ---
    with dashboard_tab:
        st.header("📈 Automated Data Dashboard")
        st.info("This dashboard provides an instant overview of your data. For deeper questions, use the chat tab.")

        # --- Data Overview & Quality ---
        st.subheader("Data Overview & Quality Metrics")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Description (Numerical):")
            st.dataframe(df.describe())
        with col2:
            st.write("Data Types & Memory Usage:")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        st.write("Missing Values:")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            st.dataframe(pd.DataFrame({"Missing Count": missing_values, "%": (missing_values / len(df) * 100).round(2)}))
        else:
            st.success("✅ No missing values found.")

        # --- Automated Plotting ---
        st.divider()
        st.subheader("Visual Column Analysis")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if numerical_cols:
            st.write("#### Numerical Distributions")
            selected_num_col = st.selectbox("Select a numerical column for detailed view:", numerical_cols)
            if selected_num_col:
                c1, c2 = st.columns(2)
                with c1:
                    fig_hist = px.histogram(df, x=selected_num_col, title=f"Histogram of {selected_num_col}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with c2:
                    fig_box = px.box(df, y=selected_num_col, title=f"Box Plot of {selected_num_col}")
                    st.plotly_chart(fig_box, use_container_width=True)
        
        if categorical_cols:
            st.write("#### Categorical Distributions")
            selected_cat_col = st.selectbox("Select a categorical column for detailed view:", categorical_cols)
            if selected_cat_col:
                if df[selected_cat_col].nunique() <= 50:
                    fig = px.histogram(df, x=selected_cat_col, title=f"Counts of {selected_cat_col}").update_xaxes(categoryorder="total descending")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption(f"Skipped '{selected_cat_col}': Has > 50 unique values.")
        
        st.divider()
        st.subheader("Relationships Between Variables")
        if len(numerical_cols) >= 2:
            fig = px.imshow(df[numerical_cols].corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Correlation heatmap requires at least two numerical columns.")

    # --- Chat Tab (from Script 2) ---
    with chat_tab:
        st.header("💬 Chat with your Data")
        st.info("I can answer questions, write Python code to analyze your data, and generate new insights. I am aware of any cleaning changes you make in the sidebar.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if isinstance(message["content"], str):
                    st.markdown(message["content"])
                else: # Handle the structured dictionary content
                    if "explanation" in message["content"]: st.markdown(message["content"]["explanation"])
                    if "code" in message["content"]: st.code(message["content"]["code"], language="python")
                    if "output" in message["content"]:
                        output = message["content"]["output"]
                        if isinstance(output, pd.DataFrame): st.dataframe(output)
                        elif isinstance(output, plt.Figure): st.pyplot(output)
                        elif output is not None: st.text(str(output))

        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar to enable the chat.")
        elif prompt := st.chat_input("Ask me to plot, analyze, or transform your data..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

# --- Main Logic Block for Processing User Prompts ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    stats.update(total_queries=stats['total_queries'] + 1)
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking, classifying, and writing code..."):
            try:
                llm = ChatGroq(model_name=model_name, groq_api_key=groq_api_key, temperature=0.1)
                
                # Classify query for stats
                try:
                    query_category = llm.invoke(CATEGORIZER_PROMPT.format(prompt=prompt)).content.strip()
                    stats['query_types'][query_category] = stats['query_types'].get(query_category, 0) + 1
                except Exception:
                    stats['query_types']['Other'] = stats['query_types'].get('Other', 0) + 1

                # Generate main response
                history_for_prompt = "\n".join([f"{m['role']}: {m['content'] if isinstance(m['content'], str) else m['content'].get('explanation','')}" for m in st.session_state.messages[:-1]])
                full_prompt = AGENT_PROMPT.format(history=history_for_prompt, prompt=prompt)
                
                llm_start_time = time.time()
                response_content = llm.invoke(full_prompt).content
                stats['total_llm_time'] += (time.time() - llm_start_time)

                explanation, code_to_execute = parse_response(response_content)

            except Exception as e:
                st.error(f"An error occurred with the LLM call: {e}")
                explanation, code_to_execute = "I encountered an error trying to generate a response.", None

        assistant_message_content = {}
        if explanation:
            st.markdown(explanation)
            assistant_message_content["explanation"] = explanation

        if code_to_execute:
            st.code(code_to_execute, language="python")
            assistant_message_content["code"] = code_to_execute

            with st.spinner("🏃‍♀️ Executing code..."):
                exec_start_time = time.time()
                output, new_df = execute_code(code_to_execute, st.session_state.current_df)
                stats['total_exec_time'] += (time.time() - exec_start_time)

                if isinstance(output, str) and output.startswith("Error"):
                    stats['failed_executions'] += 1
                    st.error(output)
                else:
                    stats['successful_executions'] += 1
                    if output is not None:
                        if isinstance(output, pd.DataFrame): st.dataframe(output)
                        elif isinstance(output, plt.Figure): st.pyplot(output)
                        else: st.text(str(output))

            st.session_state.current_df = new_df
            assistant_message_content["output"] = output

        if assistant_message_content:
            st.session_state.messages.append({"role": "assistant", "content": assistant_message_content})
            st.rerun()
        else:
            st.warning("The assistant could not generate a response. Please try again.")
            stats['failed_executions'] += 1
            st.rerun()