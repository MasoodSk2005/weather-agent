import os
import json
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types

from data_loader import WeatherDataLoader
from tools import ToolRegistry


class WeatherAgent:
    """Multi-step agent using Gemini 2.5 Flash for tool selection and orchestration."""

    def __init__(self, data_loader: WeatherDataLoader, api_key: Optional[str] = None):
        self.data_loader = data_loader
        self.tool_registry = ToolRegistry(data_loader, api_key=api_key)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_id = "gemini-3.6-flash"
        self.execution_trace: List[Dict[str, Any]] = []

    def _get_system_instruction(self) -> str:
        """System prompt enforcing tool selection, grounding, and step execution rules."""
        available_cities = ", ".join(self.data_loader.list_available_cities())
        return f"""You are an elite Weather AI Systems Engineer and Intelligent Analyst.
Your task is to answer user queries accurately using the local weather dataset and provided tools.

STRICT GROUNDING RULES:
1. Local Database Cities: [{available_cities}].
2. If a city requested by the user is NOT in this database list (e.g., Mumbai, Tokyo), DO NOT invent or hallucinate data. Instantly inform the user that records for that city are unavailable in the local dataset and state which cities ARE available.
3. Use the `calculator_tool` for ALL numeric calculations (e.g., differences, percentage changes, averages). Do NOT perform mental arithmetic.
4. Use `text_summarizer_tool` when asked to summarize or provide high-level insights on retrieved weather outputs.

WORKFLOW PROTOCOL:
- Break complex multi-part queries into sequential tool calls (Tool A -> Tool B -> Calculator -> Summarizer).
- Provide a clear, polite, and well-structured final answer after all tool steps are complete.
"""

    def _dispatch_tool(self, function_name: str, args: Dict[str, Any]) -> Any:
        """Routes FunctionCall requests from Gemini to Python implementations in ToolRegistry."""
        tool_map = {
            "weather_search_tool": self.tool_registry.weather_search_tool,
            "weather_statistics_tool": self.tool_registry.weather_statistics_tool,
            "calculator_tool": self.tool_registry.calculator_tool,
            "text_summarizer_tool": self.tool_registry.text_summarizer_tool,
        }

        if function_name not in tool_map:
            return {"status": "error", "message": f"Unknown tool: '{function_name}'"}

        try:
            return tool_map[function_name](**args)
        except Exception as e:
            return {"status": "error", "message": f"Tool execution error: {str(e)}"}

    def run(self, user_query: str, max_steps: int = 8) -> Dict[str, Any]:
        """Executes the multi-step agent loop with state tracking and function call resolution."""
        self.execution_trace = []
        
        # Pass Python functions directly to Gemini config
        config = types.GenerateContentConfig(
            system_instruction=self._get_system_instruction(),
            tools=[
                self.tool_registry.weather_search_tool,
                self.tool_registry.weather_statistics_tool,
                self.tool_registry.calculator_tool,
                self.tool_registry.text_summarizer_tool,
            ],
            temperature=0.1,
        )

        chat = self.client.chats.create(model=self.model_id, config=config)

        # Send initial user query
        response = chat.send_message(user_query)
        step_count = 0

        while step_count < max_steps:
            # Check if Gemini returned Function Calls
            if response.function_calls:
                for call in response.function_calls:
                    step_count += 1
                    fn_name = call.name
                    fn_args = dict(call.args) if call.args else {}

                    # Execute tool locally
                    tool_output = self._dispatch_tool(fn_name, fn_args)

                    # State Tracking: Record step in trace
                    trace_entry = {
                        "step": step_count,
                        "tool_selected": fn_name,
                        "inputs": fn_args,
                        "output": tool_output,
                    }
                    self.execution_trace.append(trace_entry)

                    # Return tool response back to Gemini chat history
                    response = chat.send_message(
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"result": tool_output},
                        )
                    )
            else:
                # No more function calls; final answer generated
                break

        final_text = response.text if response.text else "Query completed."
        return {
            "query": user_query,
            "final_response": final_text,
            "trace": self.execution_trace,
        }

    def print_execution_trace(self) -> None:
        """Renders clean, human-readable execution trace to console."""
        print("\n" + "=" * 70)
        print("                 AGENT EXECUTION TRACE & STATE LOG                   ")
        print("=" * 70)

        if not self.execution_trace:
            print("No external tools were invoked for this query.")
            return

        for entry in self.execution_trace:
            print(f"\n[STEP {entry['step']}] Tool Selected: {entry['tool_selected']}")
            print(f"├─ Arguments: {json.dumps(entry['inputs'])}")
            print(f"└─ Result:    {json.dumps(entry['output'], indent=2)}")

        print("=" * 70 + "\n")