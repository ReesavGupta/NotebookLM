from langchain_ollama import OllamaLLM
from PIL import Image
import base64
from io import BytesIO
import os
from langchain_groq import ChatGroq
from pydantic.types import SecretStr

llm = OllamaLLM(model="gemma3:4b") 

def encode_image_to_base64(image_path):
    img = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def multimodal_query(text_chunks, image_paths, query):
    prompt = query + "\n\n" + "\n\n".join(text_chunks)
    images_b64 = [encode_image_to_base64(p) for p in image_paths]
    response = llm.invoke(prompt, images=images_b64)
    return response

# --- Groq LLM for Query Decomposition ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_llm = ChatGroq(
        api_key=SecretStr(GROQ_API_KEY),
        model="llama-3.1-8b-instant"  # or another Groq-supported model
    )
else:
    groq_llm = None

def decompose_query(query: str) -> list:
    """Decompose a complex query into sub-queries using Groq LLM."""
    if not groq_llm:
        return [query]
    prompt = (
        "Decompose the following complex research query into a list of clear, atomic sub-queries. "
        "Return only the list, one per line.\n\nQuery: " + query + "\nSub-queries:"
    )
    result = groq_llm.invoke(prompt)
    # Extract text from BaseMessage, then split into lines
    text = getattr(result, 'content', str(result))
    subqueries = [line.strip('- ').strip() for line in text.splitlines() if line.strip()]
    return subqueries if subqueries else [query]


def classify_query_modality(query: str) -> str:
    """Classify the modality of a query using Groq LLM. Returns one of: 'text', 'image', 'table', 'code'."""
    if not groq_llm:
        return "text"
    prompt = (
        "Given the following research query, classify which modality it is primarily about. "
        "Possible modalities are: text, image, table, code. "
        "Return only the modality name in lowercase, nothing else.\n\nQuery: " + query + "\nModality:"
    )
    result = groq_llm.invoke(prompt)
    text = getattr(result, 'content', str(result)).strip().lower()
    # Only allow valid modalities
    if text in ["text", "image", "table", "code"]:
        return text
    return "text"

