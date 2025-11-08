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
from typing import Tuple, Any, Optional

# --- Configuration & Initialization ---
load_dotenv()
MAX_RETRIES = 1 # Set the maximum number of times the agent can try to correct itself

# --- Core Prompts for the Custom Agent ---
def get_df_schema(df: pd.DataFrame):
    """Generates a string representation of the dataframe's schema."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

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

**MUST-KNOW Context: DataFrame Schema**
Before you do anything, you MUST understand the data you are working with. Here is the schema of the user's DataFrame (`df`):
{schema}


**How to Respond:**
1.  **Greeting & Plan:** Start with a friendly greeting. Briefly state your understanding of the user's request and your plan to address it, keeping the provided schema in mind.
2.  **Explanation (If generating code):** Describe the analysis you're about to perform.
3.  **Code Block (If generating code):** Provide the Python code to perform the analysis within a single ```python...``` block.
4.  **Interpretation & Next Steps:** Briefly explain what the user should look for in the output and suggest a logical next question or analysis.
5.  **If no code is needed:** Simply provide a helpful, conversational answer in markdown.

**Important Rules for the Code:**
- The user's data is in a pandas DataFrame called `df`.
- **For plotting, always create a figure and axes object (e.g., `fig, ax = plt.subplots()`) and perform your plotting on the `ax` object. The `fig` object will be displayed automatically.**
- **If you are creating a plot with potentially long text labels on the x-axis (like a bar chart), rotate the labels to prevent them from overlapping. For example: `plt.xticks(rotation=45, ha='right')`**
- **To ensure all plot elements are spaced nicely, use `plt.tight_layout()` before the code block concludes.**
- Do not use any Streamlit functions (`st.*`).
- Your code must be self-contained within the markdown block.

Here is the chat history for context:
{history}

**User's Request:** {prompt}
"""

CORRECTION_PROMPT = """
You are an expert Python data science debugger. The previous code you generated failed to execute.
Your task is to analyze the user's original request, the code you wrote, and the resulting error message. Then, generate a new, corrected version of the code.

**MUST-KNOW Context: DataFrame Schema**
To help you, here is the schema of the DataFrame (`df`). The error was likely due to using a wrong column name or data type.
{schema}


**Analysis Context:**
- **Original User Request:** "{prompt}"
- **Failed Code:**
```python
{code}
```
- **Error Message:** "{error}"

**Your Task:**
1.  Briefly explain the error. What was the likely cause? (e.g., "The error occurred because I tried to use a date function on a text column.")
2.  Provide the corrected Python code. The code should be in a single markdown block.

**Corrected Response:**
"""

#--- Helper Functions ---
def parse_response(response: str):
    """Parses the LLM response to separate the explanation from the code block."""
    # Corrected regex to find the python code block
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        # Get the text before the code block
        explanation = response[: code_match.start()].strip()
        return explanation, code
    else:
        # If no code block is found, return the whole response as explanation
        return response.strip(), None

def execute_code(code: str, df: pd.DataFrame) -> Tuple[Any, pd.DataFrame, Optional[str]]:
    """
    Executes the provided Python code in a controlled environment.
    Returns the output (figure, text, or dataframe), the modified dataframe, and any error.
    """
    plt.close("all") # Close any pre-existing plots
    local_vars = {"df": df.copy(), "pd": pd, "plt": plt, "np": np, "px": px}
    string_io = io.StringIO()

    try:
        # Redirect stdout to capture print statements
        with contextlib.redirect_stdout(string_io):
            exec(code, {"__builtins__": __builtins__}, local_vars)

        new_df = local_vars.get("df", df)
        output = None

        # Check for matplotlib figures
        fig_vars = [v for v in local_vars.values() if isinstance(v, plt.Figure)]
        if fig_vars:
            output = fig_vars[0]
        elif plt.get_fignums(): # Check if a plot was created with plt.plot()
            output = plt.gcf()
        else:
            # If no plot, get the stdout
            output = string_io.getvalue()
            if not output:
                # If no stdout, try to evaluate the last line of code
                # This is useful for displaying dataframes (e.g., df.head())
                try:
                    last_line = code.strip().split('\n')[-1]
                    if "=" not in last_line:
                        output = eval(last_line, {"__builtins__": __builtins__}, local_vars)
                except Exception:
                    output = "Code executed successfully with no direct output."

    except Exception as e:
        plt.close("all")
        return None, df, str(e)

    plt.close("all")
    return output, new_df, None

@st.cache_data
def to_csv(df: pd.DataFrame) -> bytes:
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode("utf-8")

# --- Page & State Configuration ---
st.set_page_config(page_title="AI Insight Engine", layout="wide")
st.title("🤖 AI Insight Engine")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "usage_stats" not in st.session_state:
    st.session_state.usage_stats = {
        "total_queries": 0, "successful_executions": 0, "failed_executions": 0,
        "query_types": {}, "total_llm_time": 0.0, "total_exec_time": 0.0,
    }

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input(
        "Groq API Key", type="password", help="Get your key from https://console.groq.com/keys",
        value=os.environ.get("GROQ_API_KEY", "")
    )
    model_name = st.text_input(
        "Select Model", 
        value = "openai/gpt-oss-20b", index=0
    )
    st.divider()

    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        # Check if it's a new file upload to reset the session
        if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.current_df = pd.read_csv(uploaded_file)
            st.session_state.messages = [{"role": "assistant", "content": "Hello! Your data is loaded. The dashboard is ready and you can chat with me below."}]
            st.session_state.usage_stats = {
                "total_queries": 0, "successful_executions": 0, "failed_executions": 0,
                "query_types": {}, "total_llm_time": 0.0, "total_exec_time": 0.0
            }
            st.rerun()

    # Preprocessing and Session Management tools appear only after data is loaded
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
            if uploaded_file:
                uploaded_file.seek(0)
                st.session_state.current_df = pd.read_csv(uploaded_file)
                st.session_state.messages = [{"role": "assistant", "content": "DataFrame and chat have been reset."}]
                st.success("Session has been reset to the original data.")
                st.rerun()

        csv_data = to_csv(st.session_state.current_df)
        st.download_button(label="Download Modified CSV", data=csv_data, file_name="cleaned_data.csv", mime="text/csv")

# --- Main Interface ---
if st.session_state.current_df is None:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to get started.")
else:
    stats = st.session_state.usage_stats
    st.header("📊 Usage Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", stats["total_queries"])
    col2.metric("✅ Success", stats["successful_executions"])
    col2.metric("❌ Failed", stats["failed_executions"])
    total_executions = stats["successful_executions"] + stats["failed_executions"]
    avg_llm_time = (stats["total_llm_time"] / stats["total_queries"]) if stats["total_queries"] > 0 else 0
    avg_exec_time = (stats["total_exec_time"] / total_executions) if total_executions > 0 else 0
    col3.metric("Avg LLM Time", f"{avg_llm_time:.2f}s")
    col4.metric("Avg Exec Time", f"{avg_exec_time:.2f}s")

    if stats["query_types"]:
        with st.expander("Query Type Distribution"):
            df_query_types = pd.DataFrame.from_dict(stats["query_types"], orient="index", columns=["count"])
            st.bar_chart(df_query_types)

    st.divider()

    df = st.session_state.current_df
    dashboard_tab, chat_tab = st.tabs(["📊 Automated Dashboard", "💬 Chat with Data"])

    with dashboard_tab:
        st.header("📈 Automated Data Dashboard")
        st.info("This dashboard provides an instant overview of your data. For deeper questions, use the chat tab.")
        st.subheader("Data Overview & Quality Metrics")
        st.dataframe(df.head())
        col1, col2 = st.columns(2)
        with col1:
            st.write("Data Description (Numerical):")
            st.dataframe(df.describe())
        with col2:
            st.write("Data Types & Memory Usage:")
            st.text(get_df_schema(df))
        st.write("Missing Values:")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            st.dataframe(pd.DataFrame({"Missing Count": missing_values, "%": (missing_values / len(df) * 100).round(2)}))
        else:
            st.success("✅ No missing values found.")
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

    with chat_tab:
        st.header("💬 Chat with your Data")
        st.info("I can answer questions, write Python code, and even correct my own mistakes.")
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if isinstance(message["content"], str):
                    st.markdown(message["content"])
                else: # Handle dictionary content for assistant messages
                    if "explanation" in message["content"]: st.markdown(message["content"]["explanation"])
                    if "code" in message["content"]: st.code(message["content"]["code"], language="python")
                    if "error" in message["content"]: st.error(f"🚨 **Execution Error:**\n\n```\n{message['content']['error']}\n```")
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
# This block runs only when the last message is from the user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    stats["total_queries"] += 1

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking, classifying, and writing code..."):
            try:
                llm = ChatGroq(model_name=model_name, groq_api_key=groq_api_key, temperature=0.1)
                df_schema = get_df_schema(st.session_state.current_df)

                # Categorize the user query
                try:
                    query_category = llm.invoke(CATEGORIZER_PROMPT.format(prompt=prompt)).content.strip()
                    stats['query_types'][query_category] = stats['query_types'].get(query_category, 0) + 1
                except Exception:
                    stats['query_types']['Other'] = stats['query_types'].get('Other', 0) + 1

                # Format chat history for the prompt
                history_for_prompt = "\n".join([
                    f"{m['role']}: {m['content'] if isinstance(m['content'], str) else m['content'].get('explanation','')}"
                    for m in st.session_state.messages[:-1]
                ])
                full_prompt = AGENT_PROMPT.format(history=history_for_prompt, prompt=prompt, schema=df_schema)

                # Get the initial response from the LLM
                llm_start_time = time.time()
                response_content = llm.invoke(full_prompt).content
                stats['total_llm_time'] += (time.time() - llm_start_time)

                explanation, code_to_execute = parse_response(response_content)

            except Exception as e:
                st.error(f"An error occurred with the LLM call: {e}")
                explanation, code_to_execute = "I encountered an error trying to generate a response.", None

        # Prepare the message content to be stored
        assistant_message_content = {}
        if explanation:
            st.markdown(explanation)
            assistant_message_content["explanation"] = explanation

        if code_to_execute:
            st.code(code_to_execute, language="python")
            assistant_message_content["code"] = code_to_execute

            # Loop for execution and self-correction
            for i in range(MAX_RETRIES + 1):
                spinner_text = "🏃‍♀️ Executing code..." if i == 0 else f"🤔 Execution failed. Attempting to self-correct (Attempt {i}/{MAX_RETRIES})..."
                with st.spinner(spinner_text):
                    exec_start_time = time.time()
                    output, new_df, error = execute_code(code_to_execute, st.session_state.current_df)
                    stats['total_exec_time'] += (time.time() - exec_start_time)

                    if not error: # Success case
                        stats['successful_executions'] += 1
                        st.session_state.current_df = new_df
                        if output is not None:
                            if isinstance(output, pd.DataFrame): st.dataframe(output)
                            elif isinstance(output, plt.Figure): st.pyplot(output)
                            else: st.text(str(output))
                        assistant_message_content["output"] = output
                        break # Exit the correction loop on success
                    
                    # Error case
                    assistant_message_content["error"] = error
                    st.error(f"🚨 **Execution Error:**\n\n```\n{error}\n```")
                    
                    if i < MAX_RETRIES:
                        # Attempt to self-correct
                        correction_request = CORRECTION_PROMPT.format(prompt=prompt, code=code_to_execute, error=error, schema=df_schema)
                        llm_start_time = time.time()
                        correction_response = llm.invoke(correction_request).content
                        stats['total_llm_time'] += (time.time() - llm_start_time)
                        
                        explanation, code_to_execute = parse_response(correction_response)
                        
                        st.warning("⚠️ **Self-Correction Attempt**")
                        if explanation: st.markdown(explanation)
                        if code_to_execute: st.code(code_to_execute, language="python")
                        # Update message content with the corrected attempt
                        assistant_message_content["explanation"] = explanation
                        assistant_message_content["code"] = code_to_execute
                    else:
                        # Max retries reached
                        stats['failed_executions'] += 1
                        st.error("I'm sorry, I couldn't fix the code after multiple attempts. Please try rephrasing your request.")
                        break # Exit loop after final failure

        # Save the final assistant message to session state and rerun
        if assistant_message_content:
            st.session_state.messages.append({"role": "assistant", "content": assistant_message_content})
            st.rerun()
        else:
            # Handle cases where the LLM fails to generate any response
            st.warning("The assistant could not generate a response. Please try again.")
            stats['failed_executions'] += 1
            if "user" in [m["role"] for m in st.session_state.messages]: # prevent loop
                st.session_state.messages.append({"role": "assistant", "content": "I was unable to generate a response."})
            st.rerun()
