# Multi-Step AI Agent Workflow

A production-ready, pure-Python orchestration agent built from scratch using the `google-genai` SDK and Gemini 2.5 Flash. This project demonstrates ReAct execution logic, dynamic tool selection, strict dataset grounding, and safe AST-based mathematical evaluation.

## Features
- **Deterministic Multi-Step Execution:** Breaks complex queries into chained function calls.
- **Strict Grounding:** Prevents LLM hallucinations by cross-referencing requests against a local dataset, handling unsupported data inputs gracefully.
- **AST Safe Calculator:** Replaces unsafe `eval()` execution with a robust syntax-tree parser.
- **Execution Trace State Tracking:** Outputs a granular console trace detailing inputs, arguments, and intermediate step returns.

## Tech Stack
- **Language:** Python
- **LLM Engine:** Google AI Studio (`google-genai`, `gemini-2.5-flash`)
- **Data Layer:** Pandas

## Setup & Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Create a .env file in the root directory and add your API key
    GEMINI_API_KEY=your_actual_key_here

3. Run the interactive CLI:
    python main.py
