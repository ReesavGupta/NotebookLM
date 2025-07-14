from langchain_community.llms import Ollama
from PIL import Image
import base64
from io import BytesIO

llm = Ollama(model="gemma3:4b") 

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

