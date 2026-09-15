# FastAPI RAG Evaluator

A robust, asynchronous API service for evaluating the quality of Retrieval-Augmented Generation (RAG) responses using Google's Gemini AI. It acts as an LLM-Judge to assess **Faithfulness** (absence of hallucinations) and **Answer Relevance** (how well the answer addresses the user's question).

## ✨ Features

- **Automated LLM Evaluation**: Uses `gemini-2.5-flash` to score RAG responses on Faithfulness and Answer Relevance (0.0 to 1.0).
- **Strict Input Validation**: Pydantic-based validation that actively rejects common RAG failure placeholders (e.g., "no context found", "null", "нет информации").
- **Asynchronous Architecture**: Built with FastAPI and `asyncio` for high-concurrency, non-blocking API requests.
- **Automated Status Determination**: Automatically classifies evaluations as `pass` or `fail` based on configurable score thresholds (default: ≥ 0.7 for both metrics).
- **Comprehensive Testing**: Includes unit and integration tests with mocked LLM responses using `pytest` and `httpx`.
- **n8n Workflow Integration**: Ready-to-use n8n workflow templates for seamless integration into existing automation pipelines.

## 🛠️ Tech Stack

- **Framework**: FastAPI, Uvicorn
- **Data Validation**: Pydantic v2, pydantic-settings
- **AI/LLM**: Google Generative AI SDK (`google-genai`)
- **Testing**: pytest, pytest-asyncio, httpx
- **Environment**: python-dotenv

## 📂 Project Structure

```text
.
├── app/
│   ├── core/
│   │   └── config.py          # Environment and settings management
│   ├── route/
│   │   └── evaluate_route.py  # FastAPI router for evaluation endpoints
│   ├── schemas/
│   │   └── evaluation.py      # Pydantic models for request/response validation
│   ├── services/
│   │   └── judge.py           # Core LLM-Judge business logic and Gemini API integration
│   ├── main.py                # FastAPI application entry point
│   └── __init__.py
├── n8n_workflows/
│   ├── ScoredRAG.json         # n8n workflow for scoring RAG responses
│   ├── AI RAG flow.json       # n8n workflow for AI RAG processing
│   └── Load Data Flow.json    # n8n workflow for data ingestion
├── tests/
│   └── test_judge.py          # Comprehensive test suite (unit & integration)
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Pytest configuration
└── README.md                  # This file
```

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- A valid [Google Gemini API Key](https://aistudio.google.com/app/apikey)

### 2. Installation

Clone the repository and create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
PROJECT_NAME=RAG Evaluate API
```

### 4. Running the Application

Start the development server with auto-reload:

```bash
python -m app.main
```

The API will be available at:
- **Base URL**: `http://0.0.0.0:8976`
- **Interactive Swagger UI**: `http://localhost:8976/docs`

## 📡 API Endpoints

### `GET /`
Health check endpoint.
- **Response**: `{"message": "Good", "status": "running"}`

### `GET /evaluator/`
Service information endpoint.
- **Response**: `{"message": "RAG Evaluator Service is running", "status": "healthy"}`

### `POST /evaluator/evaluate`
Evaluates a RAG response based on the provided question, context, and generated answer.

**Request Body:**
```json
{
  "question": "",
  "context": "",
  "answer": ""
}


## 🧪 Testing

Run the test suite to verify the application logic, including mocked LLM responses and validation rules:

```bash
pytest tests/test_judge.py -v
```

## 🔄 n8n Integration

The `n8n_workflows/` directory contains pre-built n8n workflows to integrate this evaluator into your data pipelines:

1. **ScoredRAG.json**: A sub-workflow designed to be called by other workflows. It accepts `question`, `context`, and `answer`, sends them to the `/evaluator/evaluate` endpoint, and handles retries/errors.
2. **AI RAG flow.json**: A complete RAG pipeline example incorporating the evaluation step.
3. **Load Data Flow.json**: A workflow for ingesting and preparing data for the RAG system.

**To use**: Import the `.json` files into your n8n instance and update the `host` variable in the "Edit Fields1" node to point to your running FastAPI service (e.g., `http://localhost:8976/`).

## ⚠️ Error Handling

- **422 Unprocessable Entity**: Returned if input data fails Pydantic validation (e.g., empty context, or containing blocked placeholder phrases like "no context found").
- **504 Gateway Timeout**: Returned if the Gemini API request exceeds the 30-second timeout.
- **502 Bad Gateway**: Returned for general Google API errors.

## 📄 License

This project is provided as-is for internal evaluation and development purposes.
