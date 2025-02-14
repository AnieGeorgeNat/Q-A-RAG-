#file for testing if API key is set correctly

import google.generativeai as genai
import os

# Load API key from environment variable
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("API key not found! Set GOOGLE_API_KEY as an environment variable.")

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Test the model
response = model.generate_content("Hello, how are you?")
print(response.text)
