# AI Agent Demo Project

This project demonstrates a simple AI agent capable of reading, listing, and renaming files in a local directory using `pydantic-ai`. It supports Google Gemini, OpenAI, and DeepSeek models.

## Features

-   **Multi-Model Support**: Works with Google Gemini, OpenAI, and DeepSeek.
-   **File Operations**: Can listing, reading, and renaming files in a safe test directory.
-   **Interactive CLI**: Simple command-line interface for natural language interaction.

## Prerequisites

-   Python 3.10 or higher.
-   An API key for your chosen AI provider (Google, OpenAI, or DeepSeek).

## Setup

1.  **Clone the project**:
    Navigate to the project directory.

2.  **Create and Activate Environment**:
    You can choose between a local environment or a named global environment.

    **Option A: Local Environment (Recommended)**
    Creates a hidden `.venv` directory in the project root. Best for isolation and IDE integration.
    ```bash
    mamba env create -p ./.venv -f environment.yml
    conda activate ./.venv
    ```

    **Option B: Named Environment**
    Creates a named Conda environment (`pydantic-ai-agent`).
    ```bash
    mamba env create -f environment.yml
    conda activate pydantic-ai-agent
    ```

    **Option C: Pip (Standard Python)**
    Create a virtual environment manually and install dependencies.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install pydantic-ai python-dotenv openai google-generativeai
    ```

4.  **Configure Environment**:
    Create a `.env` file in the project root to select your provider and set your API keys.

    **Google Gemini (Default)**
    ```ini
    AI_PROVIDER=google
    GOOGLE_API_KEY=your_gemini_key
    ```

    **OpenAI**
    ```ini
    AI_PROVIDER=openai
    OPENAI_API_KEY=your_openai_key
    # Optional:
    # OPENAI_MODEL_NAME=gpt-4o
    ```

    **DeepSeek**
    ```ini
    AI_PROVIDER=deepseek
    DEEPSEEK_API_KEY=your_deepseek_key
    ```

## Usage

1.  **Run the Agent**:
    ```bash
    python main.py
    ```

2.  **Interact**:
    -   The agent operates safely within the `./test` directory.
    -   Type commands in natural language, for example:
        -   "List all files in the test folder"
        -   "Create 2D strcuture and SMILES of water molecule."

## Project Structure

-   `main.py`: Entry point of the application. Handles configuration and runs the agent loop.
-   `tools.py`: Defines the file operation tools (read, list, rename) used by the agent.
-   `environment.yml`: Defines the project dependencies.
-   `test/`: Directory where file operations are performed.
