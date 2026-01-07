import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_TEST", "AIzaSyBKI0K8jGNEho1HYERsdT8uPzaWyycwQ9g"))

print("=== LIST MODEL YANG TERSEDIA ===")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")