"""
Module To Test Out GPTs via API calls
"""
import base64
import os
import time

from openai import OpenAI

os.environ["OPENAI_API_KEY"] = (
    "<Please copy paste your opern AI key here. This program WILL NOT work otherwise>"
)


client = OpenAI()


def calculate_gpt_cost(
    inp_token: int,
    out_token: int,
    input_price: float = 0.15,  # per 1M tokens
    output_price: float = 0.60,
):  # per 1M tokens
    """
    Calculate cost for GPT-4o-mini API calls with language adjustment
    Returns:
    float: Total cost in USD
    """
    # Adjust for non-English languages (typically 2x token usage)

    input_cost = (input_price * inp_token) / 1000000
    output_cost = (output_price * out_token) / 1000000

    print(f"Input Cost: {input_cost}")
    print(f"output Cost: {output_cost}")

    return round(input_cost + output_cost, 5)


# Function to encode the image in base64 format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "Dataset/DataSetToTestGPTModels/hat_not_on_head.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)

# Starts timer to analyze the time taken by the API call
start_time = time.time()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Is the person in this image wearing a hard hat on their head? Answer just 'Hat - Yes' or 'Hat-No'",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)
end_time = time.time()

# Print Model's response
print(response.choices[0].message.content)

inp_token = response.usage.prompt_tokens
out_token = response.usage.completion_tokens

# Calculate the time taken in seconds
print(f"Time taken: {end_time - start_time:.5f} seconds")

# Calculate the cost required for this API call operation
cost = calculate_gpt_cost(inp_token, out_token)
print(f"Total cost in USD: {cost}")
