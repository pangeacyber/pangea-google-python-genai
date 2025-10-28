# Pangea + Google Gen AI SDK

A wrapper around the Google Gen AI SDK that wraps the Gemini API with
Pangea AI Guard. Supports Python v3.10 and greater.

## Installation

```bash
pip install -U pangea-google-genai
```

## Usage

```python
import os

import pangea_google_genai as genai

client = genai.PangeaClient(
    api_key=os.environ.get("GEMINI_API_KEY"),
    pangea_api_key=os.environ.get("PANGEA_API_KEY"),
    pangea_input_recipe="pangea_prompt_guard",
    pangea_output_recipe="pangea_llm_response_guard",
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)
print(response.text)
```
