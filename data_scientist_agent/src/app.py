import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import io
import contextlib
import os
import time  # <-- NEW IMPORT

# --- New Imports for Groq ---
from langchain_groq import ChatGroq

# --- MODIFICATION: New prompt for classifying user requests ---
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


def parse_response(response: str):
    code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        explanation = response[: code_match.start()].strip()
        return explanation, code
    else:
        return response.strip(), None


def execute_code(code: str, df: pd.DataFrame):
    plt.close("all")
    local_vars = {"df": df.copy(), "pd": pd, "plt": plt}
    string_io = io.StringIO()
    try:
        with contextlib.redirect_stdout(string_io):
            exec(code, {"__builtins__": __builtins__}, local_vars)
        new_df = local_vars.get("df", df)
        output = None
        fig_vars = [v for v in local_vars.values() if isinstance(v, plt.Figure)]
        if fig_vars:
            output = fig_vars[0]
        elif plt.get_fignums():
            output = plt.gcf()
        else:
            output = string_io.getvalue()
            if not output:
                try:
                    last_line = code.strip().split("\n")[-1]
                    if "=" not in last_line and not last_line.strip().startswith("df."):
                        output = eval(
                            last_line, {"__builtins__": __builtins__}, local_vars
                        )
                except Exception:
                    output = "Code executed successfully with no direct output."
        plt.close("all")
        return output, new_df
    except Exception as e:
        plt.close("all")
        return f"Error executing code: {e}", df


# --- Page Configuration ---
st.set_page_config(page_title="Data Agent Gemini", layout="wide")
st.title("🤖 Data Agent Gemini")
st.info(
    "I'm your personal AI Data Scientist. I can explain, write code, and analyze your data. "
    "Check the sidebar for live statistics on our session!"
)

# --- MODIFICATION: Session State Initialization for Usage Stats ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "usage_stats" not in st.session_state:
    st.session_state.usage_stats = {
        "total_queries": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "query_types": {},
        "total_llm_time": 0.0,
        "total_exec_time": 0.0,
    }

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get your key from https://console.groq.com/keys",
        value=os.environ.get("GROQ_API_KEY", ""),
    )
    model_name = st.selectbox(
        "Select a model",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0,
    )
    st.divider()
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        if (
            "uploaded_file_state" not in st.session_state
            or st.session_state.uploaded_file_state != uploaded_file
        ):
            st.session_state.uploaded_file_state = uploaded_file
            new_df = pd.read_csv(uploaded_file)
            st.session_state.current_df = new_df
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm ready to help. Your data is loaded.",
                }
            ]
            # Reset stats for new file
            st.session_state.usage_stats = {
                "total_queries": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "query_types": {},
                "total_llm_time": 0.0,
                "total_exec_time": 0.0,
            }
            st.rerun()
    st.divider()
    if st.session_state.current_df is not None:
        if st.button("Reset DataFrame & Chat"):
            if (
                "uploaded_file_state" in st.session_state
                and st.session_state.uploaded_file_state is not None
            ):
                st.session_state.uploaded_file_state.seek(0)
                st.session_state.current_df = pd.read_csv(
                    st.session_state.uploaded_file_state
                )
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! The chat has been reset."}
                ]
                st.success("DataFrame and chat have been reset.")
                st.rerun()

    # --- MODIFICATION: Display Usage Statistics in Sidebar ---
    st.divider()
    st.header("📊 Usage Statistics")
    stats = st.session_state.usage_stats
    col1, col2 = st.columns(2)
    col1.metric("Total Queries", stats["total_queries"])
    col2.metric("✅ Success", stats["successful_executions"])
    col2.metric("❌ Failed", stats["failed_executions"])

    avg_llm_time = (
        (stats["total_llm_time"] / stats["total_queries"])
        if stats["total_queries"] > 0
        else 0
    )
    avg_exec_time = (
        (stats["total_exec_time"] / stats["total_queries"])
        if stats["total_queries"] > 0
        else 0
    )
    st.metric("Avg LLM Time", f"{avg_llm_time:.2f}s")
    st.metric("Avg Exec Time", f"{avg_exec_time:.2f}s")

    st.write("Query Type Distribution:")
    if stats["query_types"]:
        chart_df = pd.DataFrame.from_dict(
            stats["query_types"], orient="index", columns=["count"]
        )
        st.bar_chart(chart_df)


# --- Main Chat Interface ---
if st.session_state.current_df is None:
    st.info("Please upload a CSV file in the sidebar to get started.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], str):
                st.markdown(message["content"])
            else:
                if "explanation" in message["content"]:
                    st.markdown(message["content"]["explanation"])
                if "code" in message["content"]:
                    st.code(message["content"]["code"], language="python")
                if "output" in message["content"]:
                    output = message["content"]["output"]
                    if isinstance(output, pd.DataFrame):
                        st.dataframe(output)
                    elif isinstance(output, plt.Figure):
                        st.pyplot(output)
                    elif output is not None:
                        st.text(str(output))

    if len(st.session_state.messages) == 1:
        st.header("Initial Data Overview")
        st.write("Here's a preview of the first few rows of your data:")
        st.dataframe(st.session_state.current_df.head())
        st.write("And here are the basic descriptive statistics:")
        st.dataframe(st.session_state.current_df.describe())
        st.markdown("---")

    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue.")
    elif prompt := st.chat_input("Ask me about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# --- MODIFICATION: Main logic block now collects statistics ---
if (
    st.session_state.current_df is not None
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):

    prompt = st.session_state.messages[-1]["content"]

    # --- STATS COLLECTION START ---
    st.session_state.usage_stats["total_queries"] += 1
    # --- STATS COLLECTION END ---

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking, classifying, and writing code..."):
            try:
                llm = ChatGroq(
                    model_name=model_name, groq_api_key=groq_api_key, temperature=0.1
                )

                # --- STATS COLLECTION: Classify Query ---
                try:
                    classifier_prompt = CATEGORIZER_PROMPT.format(prompt=prompt)
                    query_category = llm.invoke(classifier_prompt).content.strip()
                    stats_types = st.session_state.usage_stats["query_types"]
                    stats_types[query_category] = stats_types.get(query_category, 0) + 1
                except Exception:
                    # If classification fails, just label as "Other"
                    stats_types = st.session_state.usage_stats["query_types"]
                    stats_types["Other"] = stats_types.get("Other", 0) + 1
                # --- STATS COLLECTION END ---

                history_for_prompt = "\n".join(
                    [
                        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'] if isinstance(m['content'], str) else m['content'].get('explanation', '')}"
                        for m in st.session_state.messages[:-1]
                    ]
                )
                full_prompt = AGENT_PROMPT.format(
                    history=history_for_prompt, prompt=prompt
                )

                # --- STATS COLLECTION: Time LLM Call ---
                llm_start_time = time.time()
                response_content = llm.invoke(full_prompt).content
                llm_end_time = time.time()
                st.session_state.usage_stats["total_llm_time"] += (
                    llm_end_time - llm_start_time
                )
                # --- STATS COLLECTION END ---

                explanation, code_to_execute = parse_response(response_content)

            except Exception as e:
                # Handle LLM errors
                st.error(f"An error occurred with the LLM call: {e}")
                explanation, code_to_execute = None, None

        assistant_message_content = {}
        if explanation:
            st.markdown(explanation)
            assistant_message_content["explanation"] = explanation

        if code_to_execute:
            st.code(code_to_execute, language="python")
            assistant_message_content["code"] = code_to_execute

            with st.spinner("🏃‍♀️ Executing code..."):
                # --- STATS COLLECTION: Time Execution & Track Success/Failure ---
                exec_start_time = time.time()
                output, new_df = execute_code(
                    code_to_execute, st.session_state.current_df
                )
                exec_end_time = time.time()
                st.session_state.usage_stats["total_exec_time"] += (
                    exec_end_time - exec_start_time
                )

                if isinstance(output, str) and output.startswith(
                    "Error executing code:"
                ):
                    st.session_state.usage_stats["failed_executions"] += 1
                else:
                    st.session_state.usage_stats["successful_executions"] += 1
                # --- STATS COLLECTION END ---

            st.session_state.current_df = new_df
            if output is not None:
                if isinstance(output, pd.DataFrame):
                    st.dataframe(output)
                elif isinstance(output, plt.Figure):
                    st.pyplot(output)
                else:
                    st.text(str(output))
            assistant_message_content["output"] = output

        if assistant_message_content:
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_message_content}
            )
            st.rerun()
        else:
            st.warning("The assistant could not generate a response. Please try again.")
            st.session_state.usage_stats[
                "failed_executions"
            ] += 1  # Count as failure if no response
