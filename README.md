# Weather Agent

This notebook implements a tool-using weather assistant that combines Gemini with a local weather-data loader and a constrained calculator. The agent can interpret a weather-related question, decide when to use the available tools, retrieve information from the notebook's weather dataset, and present a response based on the returned data.


## Components

- **WeatherDataLoader:** loads and inspects the weather data with Pandas, exposes available cities, and provides city-level statistics.
- **SafeCalculator:** evaluates supported arithmetic expressions using a restricted set of operations rather than unrestricted Python execution.
- **ToolRegistry:** registers tools and makes their descriptions and execution functions available to the agent.
- **WeatherAgent:** uses the Gemini API to interpret the request, select tools, and construct a final answer. The notebook also provides an execution trace to show the agent's steps.


## Requirements

- Python
- Google GenAI SDK (`google-genai`)
- Pandas
- `python-dotenv` (optional, for `.env` configuration)
- A Gemini API key
- The weather dataset expected by the notebook's data loader


## Setup

1. Open `WeatherAgent_Gemini_Audited(2).ipynb` in Google Colab or Jupyter.
2. Run the package installation cell.
3. Set `GEMINI_API_KEY` in your environment or local `.env` file. The notebook checks for the key without displaying its value.
4. Make sure the expected weather dataset is available at the configured path.
5. Run the notebook cells in order to load data, initialize tools, and create the agent.


## Asking questions

Run the single-query cell and enter a weather question when prompted. The notebook prints the agent's execution trace and final response. The optional interactive cell keeps the session open for multiple questions; enter `exit` or `quit` to stop.


## Data and limitations

The agent can only report on locations and fields present in its loaded dataset and tools. It is not automatically a live weather service unless the underlying data source is live and configured that way. Check the dataset's date and coverage before using an answer for current conditions. Gemini can misunderstand a question or make mistakes when interpreting tool results, so verify important information against the data.
