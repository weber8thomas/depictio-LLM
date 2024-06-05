import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import os
from dotenv import load_dotenv
import langchain
from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc   

load_dotenv()

print(os.environ["OPENAI_API_KEY"])

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load a dummy dataframe
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'age': [25, 30, 35, 40, 22],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
}
df = pd.DataFrame(data)

# Create the agent
llm = OpenAI(temperature=0)
agent = create_pandas_dataframe_agent(llm, df)

# Layout of the Dash app
app.layout = html.Div([
    dmc.Title("Natural Language DataFrame Query", order=1),
    dmc.TextInput(id='user-input', placeholder='Ask a question...', style={'width': '60%'}),
    dmc.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='output-response'),
    html.Div(id='output-code')
])

# Callback to handle user input and generate the result
@app.callback(
    Output('output-response', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('user-input', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0 and value:
        try:
            # Capture the generated code
            response = agent.run(value)
            return html.Pre(str(response))
        except Exception as e:
            return html.Pre(f"Error: {e}")
    return "Enter a question and click Submit"

if __name__ == '__main__':
    app.run_server(debug=True)