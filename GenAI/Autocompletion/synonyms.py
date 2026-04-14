import argparse
import os
import sys
from typing import Optional

import requests
from dotenv import load_dotenv

# ✅ Latest OpenAI SDK
from openai import OpenAI


load_dotenv()

# -----------------------------
# ENV HANDLING
# -----------------------------
def load_env(env_path: str = ".env") -> None:
    load_dotenv(env_path)


def get_env_value(key: str, required: bool = False) -> str:
    value = os.getenv(key, "").strip()
    if required and not value:
        raise ValueError(f"{key} is required in .env")
    return value


# -----------------------------
# PROMPT BUILDER
# -----------------------------
def build_prompt(text: str, limit: int = 10) -> str:
    return f"""
You are an ecommerce grocery search assistant.

Your job is to expand user queries into Hinglish search terms that help retrieve relevant products on a grocery platform.

Rules:
- Output ONLY a numbered list
- Use Hinglish (Hindi words written in English, e.g., "tamatar", "lal sabzi")
- Focus on search intent, not literal synonyms
- Include variations users might type in a search bar
- Avoid irrelevant or poetic phrases
- Keep terms short and practical for search
- Avoid adding extra words like verbs or adjectives unless they are commonly used in search queries
- Max {limit} items

User query: {text}
""".strip()


# -----------------------------
# OPENAI (LATEST SYNTAX)
# -----------------------------
def call_openai(model: str, prompt: str) -> str:
    client = OpenAI(api_key=get_env_value("OPENAI_API_KEY", True))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You generate Hindi vernacular synonyms."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


# -----------------------------
# LLAMA (GENERIC HTTP WRAPPER)
# -----------------------------
def call_llama(model: str, prompt: str) -> str:
    api_url = get_env_value("LLAMA_API_URL", True)
    api_key = get_env_value("LLAMA_API_KEY", True)

    response = requests.post(
        api_url,
        json={
            "model": model,
            "input": prompt,
            "temperature": 0.8,
            "max_output_tokens": 300,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=20,
    )

    response.raise_for_status()
    data = response.json()

    return (
        data.get("output")
        or (data.get("choices") or [{}])[0].get("text", "")
        or str(data)
    ).strip()


# -----------------------------
# GEMMA (UPDATED STYLE)
# -----------------------------
def call_gemma(model: str, prompt: str) -> str:
    api_url = get_env_value("GEMMA_API_URL", True)
    api_key = get_env_value("GEMMA_API_KEY", True)

    response = requests.post(
        api_url,
        json={
            "model": model,
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 300,
            },
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=20,
    )

    response.raise_for_status()
    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return str(data)


# -----------------------------
# OLLAMA (LOCAL LLM)
# -----------------------------
def call_ollama(model: str, prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    return data.get("response", "").strip()


# -----------------------------
# ROUTER
# -----------------------------
def get_default_model(provider: str) -> str:
    return {
        "openai": "gpt-4o-mini",
        "llama": "llama3",
        "gemma": "gemma:2b",
    }.get(provider, "gpt-4o-mini")


def get_synonyms(
    text: str,
    limit: int,
    provider: str,
    model: Optional[str],
) -> str:
    prompt = build_prompt(text, limit)
    model = model or get_default_model(provider)

    if provider == "openai":
        return call_openai(model, prompt)

    # elif provider == "llama":
    #     return call_llama(model, prompt)

    # elif provider == "gemma":
    #     return call_gemma(model, prompt)
    
    elif provider in ["llama", "gemma"]:
        return call_ollama(model, prompt)    

    else:
        raise ValueError("Invalid provider")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Hindi vernacular synonyms using LLMs"
    )

    parser.add_argument("text", nargs="+")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--provider",
        choices=["openai", "llama", "gemma"],
        default="openai",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--env", default=".env")

    return parser.parse_args()


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    args = parse_args()

    try:
        load_env(args.env)

        input_text = " ".join(args.text)

        result = get_synonyms(
            text=input_text,
            limit=args.limit,
            provider=args.provider,
            model=args.model,
        )

        print(result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()