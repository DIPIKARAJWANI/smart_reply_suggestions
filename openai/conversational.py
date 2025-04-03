import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langdetect import detect
import os
from dotenv import load_dotenv
import openai
import json

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit(1)

# Load FAISS index and dataset
try:
    index = faiss.read_index("faiss_index.bin")
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    exit(1)

df_products = pd.read_csv("indexed_data.csv")
print(f"Loaded {len(df_products)} products from indexed_data.csv.")

# Load website information
try:
    with open("data/website.json", "r", encoding="utf-8") as f:
        website_data = json.load(f)
    print("Website info loaded successfully.")
except Exception as e:
    print(f"Error loading website.json: {e}")
    exit(1)

# Load sentence transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def detect_language(query):
    """Detects query language."""
    try:
        lang = detect(query)
        return lang if lang in ["en", "hi", "gu"] else "en"
    except:
        return "en"

def search_faiss(query):
    """Searches FAISS for context."""
    query_vector = model.encode(query).astype("float32")
    distances, indices = index.search(np.array([query_vector]), k=1)
    return df_products.iloc[indices[0][0]].to_dict() if indices[0][0] != -1 else None

def generate_replies(query):
    """Generates three dynamic, human-like replies using GPT-4."""
    lang = detect_language(query)
    context = search_faiss(query)
    
    # Map language codes to names
    lang_map = {"en": "English", "hi": "Hindi", "gu": "Gujarati"}
    lang_name = lang_map.get(lang, "English")
    
    # Prepare context for GPT-4
    context_str = ""
    if context:
        context_str = f"Relevant product: {context['name']} ({context['price']}, {context['availability']}). "
    policy_str = f"Refund policy: {website_data.get('company', {}).get('refund_policy', {}).get('summary', 'Returns possible within a set period—email us.')}"
    contact_str = f"Contact: {website_data.get('company', {}).get('contact', {}).get('email', 'support@2xnutrition.com')}"

    prompt = (
        f"Generate exactly 3 short, human-like replies to '{query}' in {lang_name}. "
        f"Each reply should be 5-10 words, natural, engaging, and unique. "
        f"Use this context only if relevant: {context_str}{policy_str}. {contact_str}. "
        f"Focus on conversation, match the query intent, avoid product pushing unless natural."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.8
        )
        replies = response.choices[0].message.content.strip().split("\n")
        # Ensure 3 replies
        while len(replies) < 3:
            replies.append("Let’s chat more—what’s your next step?")
        return replies[:3]
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return [
            "Oops, something broke—try again?",
            "Yikes, error—retry your query?",
            "Uh-oh, glitch—ask me again?"
        ]

def main():
    print("FAISS index loaded successfully.")
    print("Website info loaded successfully.\n")
    
    while True:
        query = input("Enter customer query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        replies = generate_replies(query)
        
        print("\nSmart Reply Suggestions:")
        for i, reply in enumerate(replies, 1):
            print(f"{i}. {reply}")
        print()

if __name__ == "__main__":
    main()