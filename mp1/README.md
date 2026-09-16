# MP1: Prompt Lab

MP1 explores prompt strategies for extracting structured information from job
postings. The project uses the OpenAI API, Pydantic models, and `uv` for
dependency management.

## Prerequisites

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI API key with access to the configured model

## Setup

From the repository root:

```bash
cd mp1
uv sync
```

Create a `.env` file inside `mp1/`:

```env
OPENAI_API_KEY=your-api-key-here
```

Do not commit `.env` or share the API key.