import sys

from openai import OpenAI

client = OpenAI()


def ask(question: str) -> str:
    """Send a question to the LLM and return the response."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are concise.",
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        temperature=0.3,
    )
    return response.output_text


if __name__ == "__main__":
    # question = input("Ask a question: ")
    question = q = " ".join(sys.argv[1:]) or "Say hello in one sentence."
    answer = ask(question)
    print(f"Answer: {answer}")
