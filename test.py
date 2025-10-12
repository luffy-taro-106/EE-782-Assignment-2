from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
load_dotenv()

# Set your API key
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


# Initialize the Gemini client
client = genai.Client()

while True:
    query = input("Enter your query for Gemini (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # disables thinking
            ),
        )
        print("Gemini says:", response.text)
    except Exception as e:
        print("Error communicating with Gemini:", e)
