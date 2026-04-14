import time
from dotenv import load_dotenv
from synonyms import get_synonyms

TEST_QUERIES = [
    "tomato",
    "milk",
    "chips",
    "atta",
    "cold drink",
]

PRICING = {
    "openai": {
        "gpt-4o-mini": 0.003 / 1000,
        "gpt-4o": 0.006 / 1000,
        "gpt-3.5-turbo": 0.0015 / 1000,
    },
    "llama": {},
    "gemma": {},
}


def get_model_cost(provider: str, model: str, total_tokens):
    if total_tokens is None or not isinstance(total_tokens, int):
        return None
    per_token = PRICING.get(provider, {}).get(model)
    if per_token is None:
        return None
    return total_tokens * per_token


def format_table(rows, headers):
    widths = [max(len(str(item)) for item in column) for column in zip(headers, *rows)]
    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    separator = "-+-".join("-" * width for width in widths)
    row_lines = [" | ".join(str(item).ljust(width) for item, width in zip(row, widths)) for row in rows]
    return "\n".join([header_line, separator] + row_lines)


def benchmark(provider, model=None):
    print(f"\n--- {provider.upper()} BENCHMARK ---")
    rows = []
    total_time = 0.0
    total_cost = 0.0
    total_tokens = 0
    token_rows = 0

    for query in TEST_QUERIES:
        start = time.time()
        result = get_synonyms(
            text=query,
            limit=10,
            provider=provider,
            model=model,
        )
        latency = time.time() - start
        total_time += latency

        usage = result.get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = usage.get("prompt_token")
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = usage.get("completion_token")

        total_tokens_used = usage.get("total_tokens")
        if total_tokens_used is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens_used = prompt_tokens + completion_tokens

        cost = get_model_cost(provider, result.get("model"), total_tokens_used)
        if isinstance(total_tokens_used, int):
            total_tokens += total_tokens_used
            token_rows += 1
        if cost is not None:
            total_cost += cost

        rows.append([
            provider,
            result.get("model", "N/A"),
            query,
            prompt_tokens if prompt_tokens is not None else "N/A",
            completion_tokens if completion_tokens is not None else "N/A",
            total_tokens_used if total_tokens_used is not None else "N/A",
            f"${cost:.6f}" if cost is not None else "N/A",
            f"{latency:.2f}s",
        ])

    print(format_table(
        rows,
        [
            "Provider",
            "Model",
            "Query",
            "Prompt Tokens",
            "Completion Tokens",
            "Total Tokens",
            "Cost",
            "Latency",
        ],
    ))

    average_latency = total_time / len(TEST_QUERIES)
    print(f"\nAverage latency: {average_latency:.2f}s")
    if token_rows:
        print(f"Total tokens (recorded): {total_tokens}")
    if total_cost:
        print(f"Total cost: ${total_cost:.6f}")
    print()


if __name__ == "__main__":
    load_dotenv()
    benchmark("openai")
    benchmark("llama")
    # benchmark("gemma")
