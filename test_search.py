import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

models_to_try = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-3.5-flash", 
    "gemini-3.5-pro", 
    "gemini-3.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

for model_name in models_to_try:
    print(f"Trying model: {model_name}")
    try:
        search_model = genai.GenerativeModel(
            model_name=model_name,
            tools="google_search_retrieval"
        )
        search_response = search_model.generate_content("今日の東京の天気を教えて")
        print(f"Success with {model_name}")
        break
    except Exception as e:
        print(f"Error with {model_name}: {type(e).__name__} - {e}")
