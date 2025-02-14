from dotenv import load_dotenv
import os

# Try to load .env from both project root and backend directory
project_root = os.path.dirname(os.path.dirname(__file__))
backend_dir = os.path.dirname(__file__)

# Try loading from project root first, then backend
load_dotenv(os.path.join(project_root, '.env'))
load_dotenv(os.path.join(backend_dir, '.env'))

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY is missing. Set it in the .env file.")

def get_genai_api_key():
    return GENAI_API_KEY

from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embedding model
embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")    