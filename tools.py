import ast
import operator
import os
from typing import Dict, Any, List, Optional, Union
from google import genai
from data_loader import WeatherDataLoader


class SafeCalculator:
    """Evaluates mathematical expressions safely using Python's AST module."""

    # Supported safe operators
    _ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def evaluate(cls, expression: str) -> Union[float, int, str]:
        """Parses and safely computes a mathematical expression string."""
        try:
            node = ast.parse(expression.strip(), mode="eval").body
            return round(cls._eval_node(node), 4)
        except ZeroDivisionError:
            return "Error: Division by zero."
        except Exception as e:
            return f"Error: Invalid mathematical expression '{expression}'. ({str(e)})"

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Union[float, int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            op_type = type(node.op)
            if op_type in cls._ALLOWED_OPERATORS:
                return cls._ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in cls._ALLOWED_OPERATORS:
                return cls._ALLOWED_OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        else:
            raise ValueError(f"Unsupported expression construct: {type(node).__name__}")


class ToolRegistry:
    """Central registry providing executable interfaces and schemas for the LLM agent."""

    def __init__(self, data_loader: WeatherDataLoader, api_key: Optional[str] = None):
        self.loader = data_loader
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._genai_client = None

    def _get_client(self) -> genai.Client:
        """Lazy initialization of Google GenAI client."""
        if self._genai_client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable is missing. "
                    "Set it or pass it to ToolRegistry."
                )
            self._genai_client = genai.Client(api_key=self.api_key)
        return self._genai_client

    # --- Tool 1: Weather Search ---
    def weather_search_tool(
        self, city: str, date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Queries raw weather entries for a specific city and optional date."""
        records = self.loader.get_city_weather(city, date)
        if not records:
            available = self.loader.list_available_cities()
            return {
                "status": "not_found",
                "message": f"No weather records found for '{city}'"
                + (f" on date '{date}'" if date else "")
                + ".",
                "grounding_info": f"Available cities in database: {', '.join(available)}",
            }
        return {"status": "success", "count": len(records), "data": records}

    # --- Tool 2: Weather Statistics ---
    def weather_statistics_tool(self, city: str) -> Dict[str, Any]:
        """Calculates aggregations (average temp, max/min temp, total rain, humidity)."""
        stats = self.loader.get_city_stats(city)
        if not stats.get("found", False):
            available = self.loader.list_available_cities()
            return {
                "status": "not_found",
                "message": stats.get("message", f"City '{city}' not found."),
                "grounding_info": f"Available cities in database: {', '.join(available)}",
            }
        return {"status": "success", "statistics": stats}

    # --- Tool 3: Safe Calculator ---
    def calculator_tool(self, expression: str) -> Dict[str, Any]:
        """Safely evaluates arithmetic expressions (e.g., '28.5 - 14.2' or '(10 + 5) / 2')."""
        result = SafeCalculator.evaluate(expression)
        if isinstance(result, str) and result.startswith("Error:"):
            return {"status": "error", "message": result}
        return {"status": "success", "expression": expression, "result": result}

    # --- Tool 4: Text Summarizer ---
    def text_summarizer_tool(
        self, content: str, focus: str = "general summary"
    ) -> Dict[str, Any]:
        """Uses Gemini 2.5 Flash to generate a concise summary of intermediate outputs."""
        try:
            client = self._get_client()
            prompt = (
                f"You are a weather intelligence assistant. Summarize the following data concisely, "
                f"focusing on '{focus}':\n\n{content}"
            )
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            return {
                "status": "success",
                "focus": focus,
                "summary": response.text.strip(),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to generate summary: {str(e)}",
            }