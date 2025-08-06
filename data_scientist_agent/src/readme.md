# Colab-Style Data Scientist Agent

This Streamlit application provides an interactive, chat-based interface to a powerful AI data scientist. The agent can understand plain English commands, write Python code to analyze a pandas DataFrame, execute the code, and display the results (tables, plots, values) directly in the app.

## Features

- **Code Generation:** The agent writes Python code to perform data analysis tasks.
- **Code Execution:** The generated code is executed in a secure environment.
- **Stateful DataFrame:** Changes made to the DataFrame (e.g., dropping columns, filling missing values) are persisted across conversational turns.
- **Interactive Visualization:** The agent can generate and display plots using Plotly and Matplotlib.
- **Modular & Scalable:** The project is structured with a clear separation of concerns, making it easy to extend.

## File Structure
data-scientist-agent/
|
├── .env
├── README.md
├── requirements.txt
|
├── data/
│ └── titanic.csv
|
└── src/
├── init.py
├── app.py
├── agent.py
└── code_executor.py

## Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd data-scientist-agent
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    -   Create a file named `.env` in the root directory (`data-scientist-agent/`).
    -   Add your Google Generative AI API key to it:
        ```
        GOOGLE_API_KEY="your_google_api_key_here"
        ```

5.  **Data:**
    -   Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data).
    -   Place the `train.csv` file inside the `data/` folder and rename it to `titanic.csv`.

## Running the Application

Navigate to the project's root directory in your terminal and run the following command:

```bash
streamlit run src/app.py
The application will open in your web browser.
How to Use
Upload a CSV file using the sidebar.
Once the data is loaded, start asking the agent questions in the chat input box.
Example Prompts:
Show the first 10 rows
What are the data types of each column?
Fill the missing values in the 'Age' column with the median.
Plot a bar chart of the 'Sex' column.
Show me the correlation matrix as a heatmap.