import requests
import json
import os
from dotenv import load_dotenv


def make_a_api_call(content_query, user_role="user"):
    """
    Make a call to the OpenRouter API and print the response.
    """

    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    MODEL = os.getenv("MODEL")
    OPENROUTER_URL = os.getenv("OPENROUTER_URL")

    # Check if the API key is set
    if OPENROUTER_API_KEY is None:
        raise ValueError(
            "OPENROUTER_API_KEY is not set in the environment variables.")

    # Check if the model is set
    if MODEL is None:
        raise ValueError("MODEL is not set in the environment variables.")
    # Check if the URL is set
    if OPENROUTER_URL is None:
        raise ValueError(
            "OPENROUTER_URL is not set in the environment variables.")
    # Check if the user role is valid
    if user_role not in ["user", "assistant"]:
        raise ValueError("user_role must be either 'user' or 'assistant'.")
    # Check if the content query is valid
    if not isinstance(content_query, str):
        raise ValueError("content_query must be a string.")
    # Check if the content query is empty
    if not content_query.strip():
        raise ValueError("content_query must not be empty.")
    # Check if the content query is too long
    if len(content_query) > 4096:
        raise ValueError(
            "content_query must not be longer than 4096 characters.")
    # Check if the content query is too short
    if len(content_query) < 1:
        raise ValueError("content_query must be at least 1 character long.")

    # Make a POST request to the OpenRouter API
    response = requests.post(
        url=OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            # Optional. Site URL for rankings on openrouter.ai.
            # "HTTP-Referer": "<YOUR_SITE_URL>",
            # Optional. Site title for rankings on openrouter.ai.
            # "X-Title": "<YOUR_SITE_NAME>",
        },
        data=json.dumps({
            "model": MODEL,  # Optional
            "messages": [
                {
                    "role": user_role,
                    "content": content_query
                }
            ],
            # Token limiting parameters kolla med modellen om detta är ok:
            "max_tokens": 100,          # Maximum number of tokens in the response
            # Controls randomness (not length related, but affects output)
            "temperature": 0.7,
            "stream": False             # Whether to stream the response tokens
        })
    )

    return response


# test att API anropet fungerar
if __name__ == "__main__":
    # Example usage
    content_query = "What is the capital of France?"
    user_role = "user"  # or "assistant"
    response = make_a_api_call(content_query, user_role)
    print(response.status_code)
    print(response.json())
    # Check if the response is successful
    if response.status_code == 200:
        # Parse the JSON response
        response_json = response.json()
        # Print the response
        print(json.dumps(response_json, indent=4))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
