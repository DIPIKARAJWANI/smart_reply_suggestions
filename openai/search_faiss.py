# import faiss
# import numpy as np
# import pandas as pd
# import os
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI

# # Load FAISS index and dataset
# index = faiss.read_index("faiss_index.bin")
# df = pd.read_csv("indexed_data.csv")

# # Load embedding model
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def generate_smart_replies(query, top_k=3):
#     """Retrieve relevant data from FAISS and generate 3 short, formal replies."""
#     query_embedding = model.encode([query]).astype("float32")
#     D, I = index.search(query_embedding, top_k)
#     results = df.iloc[I[0]]

#     context = "\n".join(results["description"].dropna().tolist())

#     if not context:
#         return ["I'm sorry, but I couldn't find relevant information."]

#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "You are a professional e-commerce assistant. Respond concisely and formally with exactly three suggestions. Each response should be short, professional, and clear."},
#             {"role": "user", "content": f"Customer Query: {query}\n\nRelevant Info: {context}\n\nProvide exactly three short and formal reply suggestions, separated by new lines."}
#         ],
#         temperature=0.5,
#         max_tokens=100,  # Allow enough space for 3 replies
#         n=1
#     )

#     # Split responses properly and clean formatting
#     replies = [line.strip("-•1234567890. ") for line in response.choices[0].message.content.split("\n") if line.strip()]
    
#     return replies[:3] if len(replies) >= 3 else ["I'm sorry, I couldn't generate 3 responses."]

# # Example query
# query = "how do i gain weight?"
# print(query)

# suggestions = generate_smart_replies(query)

# print("\nSmart Reply Suggestions:")
# for i, suggestion in enumerate(suggestions, 1):
#     print(f"{i}. {suggestion}")

# import faiss
# import numpy as np
# import json
# from langdetect import detect
# from openai import OpenAI
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util

# # Load FAISS index and dataset
# try:
#     index = faiss.read_index("faiss_index.bin")
#     print("FAISS index loaded successfully.")
# except Exception as e:
#     print("Error loading FAISS index:", e)
#     exit(1)

# df_products = pd.read_csv("indexed_data.csv")  # Product information
# print(f"Loaded {len(df_products)} products from indexed_data.csv.")

# # Load website.json
# try:
#     with open("data/website.json", "r", encoding="utf-8") as f:
#         website_data = json.load(f)
#     print("Website info loaded successfully.")
# except Exception as e:
#     print("Error loading website.json:", e)
#     exit(1)

# # OpenAI client
# client = OpenAI()

# # Load sentence transformer model for text similarity
# model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# def detect_language(query):
#     """Detects language of the query and returns language code."""
#     try:
#         lang = detect(query)
#         if lang not in ["en", "hi", "gu", "wa"]:  # Only support English, Hindi, Gujarati, WhatsApp-style language
#             return "en"  # Default to English
#         return lang
#     except:
#         return "en"

# def search_faiss(query):
#     """Searches FAISS and retrieves the most relevant product details."""
#     query_vector = model.encode(query).astype("float32")
#     _, indices = index.search(np.array([query_vector]), k=3)
#     results = [df_products.iloc[i].to_dict() for i in indices[0] if i != -1]
    
#     # Rank results based on text similarity
#     best_match = max(results, key=lambda x: util.pytorch_cos_sim(query_vector, model.encode(x["description"])).item(), default=None)
    
#     return json.dumps(best_match, ensure_ascii=False) if best_match else ""

# def generate_replies(query, context, lang):
#     """Generates smart reply suggestions based on context and language."""
#     try:
#         # Check for restricted queries
#         restricted_queries = ["email", "contact number", "phone number", "address"]
#         if any(word in query.lower() for word in restricted_queries):
#             return ["I'm sorry, but we cannot provide that information. Please check our website for details."]
        
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": f"You are a professional e-commerce assistant. Always reply in {lang}. Keep responses short and professional."},
#                 {"role": "user", "content": f"Customer Query: {query}\n\nRelevant Info: {context}\n\nProvide exactly three concise and professional reply suggestions in {lang}."}
#             ],
#             temperature=0.2,
#             max_tokens=40,  # Even shorter responses
#             n=1
#         )

#         if response and response.choices:
#             replies = response.choices[0].message.content.split("\n")
#             replies = [r.strip("-•1234567890. ") for r in replies if r.strip()]
#         else:
#             replies = ["Sorry, I couldn't generate a response."]
#     except Exception as e:
#         print(f"Error in generating reply: {e}")
#         replies = ["I'm sorry, something went wrong. Please try again."]
    
#     return replies

# def main():
#     print("FAISS index loaded successfully.")
#     print("Website info loaded successfully.\n")
    
#     while True:
#         query = input("Enter customer query (or type 'exit' to quit): ")
#         if query.lower() == 'exit':
#             break
        
#         lang = detect_language(query)
#         context = search_faiss(query)
#         replies = generate_replies(query, context, lang)
        
#         print("\nSmart Reply Suggestions:")
#         for i, reply in enumerate(replies, 1):
#             print(f"{i}. {reply}")
#         print()

# if __name__ == "__main__":
#     main()
import faiss
import numpy as np
import json
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from langdetect import detect

# Load FAISS index and dataset
try:
    index = faiss.read_index("faiss_index.bin")
    print("FAISS index loaded successfully.")
except Exception as e:
    print("Error loading FAISS index:", e)
    exit(1)

df_products = pd.read_csv("indexed_data.csv")
print(f"Loaded {len(df_products)} products from indexed_data.csv.")

# Load website information
try:
    with open("data/website.json", "r", encoding="utf-8") as f:
        website_data = json.load(f)
    print("Website info loaded successfully.")
except Exception as e:
    print("Error loading website.json:", e)
    exit(1)

# Load sentence transformer model for text similarity
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def detect_language(query):
    """Detects language of the query."""
    try:
        lang = detect(query)
        return lang if lang in ["en", "hi", "gu", "wa"] else "en"
    except:
        return "en"

def preprocess_query(query):
    """Cleans and normalizes query text."""
    query = query.lower().strip()
    query = re.sub(r'[^a-zA-Z0-9\s]', '', query)  # Remove special characters
    return query

def search_faiss(query):
    """Searches FAISS and retrieves the most relevant product details."""
    query_vector = model.encode(query).astype("float32")
    _, indices = index.search(np.array([query_vector]), k=5)  # Get top 5 matches
    results = [df_products.iloc[i].to_dict() for i in indices[0] if i != -1]
    
    # Prioritize exact name matches
    for result in results:
        if query == result['name'].lower():
            return json.dumps(result, ensure_ascii=False)
    
    # Improve result selection by checking keyword presence in product name
    for result in results:
        if any(word in result['name'].lower() for word in query.split()):
            return json.dumps(result, ensure_ascii=False)
    
    return json.dumps(results[0], ensure_ascii=False) if results else ""

def get_website_info(category):
    """Fetch relevant website details."""
    info_map = {
        "contact": [
            f"Email: {website_data.get('company', {}).get('contact', {}).get('email', 'Not available')}",
            f"Address: {website_data.get('company', {}).get('contact', {}).get('address', 'Not available')}"
        ],
        "policy": [
            website_data.get("company", {}).get("shipping", {}).get("policy", "Shipping policy not available."),
            website_data.get("company", {}).get("refund_policy", {}).get("summary", "Refund policy not available.")
        ],
        "return": [
            "To process a return, please provide your order ID.",
            "Returns are subject to our policy. Please share your order details."
        ],
        "allergy": [
            "Check the ingredients list for potential allergens.",
            "We cannot guarantee an allergen-free product.",
            "Consult a doctor if you have allergies or concerns."
        ],
        "weight_gain": search_faiss("mass gainer"),
        "weight_loss": search_faiss("fat burner"),
        "merchandise": search_faiss("t-shirt"), # Ensure proper merchandise match
        "whey_protein": search_faiss("IsoMagic")
    }
    return info_map.get(category, [])

def generate_replies(query):
    """Generates smart reply suggestions."""
    query = preprocess_query(query)
    
    predefined_queries = {
        "email": "contact", "phone number": "contact", "address": "contact", 
        "policy": "policy", "refund": "policy", "shipping": "policy", "return": "return",
        "allergy": "allergy", "allergic": "allergy",
        "gain weight": "weight_gain", "lose weight": "weight_loss",
        "mass gainer": "weight_gain", "fat burner": "weight_loss",
        "tshirt": "merchandise", "shirt": "merchandise", "merchandise": "merchandise",
        "whey protein": "whey_protein"
    }
    
    for key, category in predefined_queries.items():
        if key in query:
            info = get_website_info(category)
            return info if isinstance(info, list) else [
                f"{json.loads(info)['name']} is available for ₹{json.loads(info)['price']}.",
                f"You can order {json.loads(info)['name']} from our website.",
                f"Check more details about {json.loads(info)['name']} on our website."
            ]
    
    # Search for product details
    product_data = search_faiss(query)
    if product_data:
        product = json.loads(product_data)
        return [
            f"{product['name']} is available for ₹{product['price']}.", 
            f"You can order {product['name']} from our website.",
            f"Check more details about {product['name']} on our website."
        ]
    
    return ["I'm here to help! Please provide more details."]

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

# import faiss
# import numpy as np
# import json
# from langdetect import detect
# from openai import OpenAI
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util

# # Load FAISS index and dataset
# try:
#     index = faiss.read_index("faiss_index.bin")
#     print("FAISS index loaded successfully.")
# except Exception as e:
#     print("Error loading FAISS index:", e)
#     exit(1)

# df_products = pd.read_csv("indexed_data.csv")  # Product information
# print(f"Loaded {len(df_products)} products from indexed_data.csv.")

# # Load website.json
# try:
#     with open("data/website.json", "r", encoding="utf-8") as f:
#         website_data = json.load(f)
#     print("Website info loaded successfully.")
# except Exception as e:
#     print("Error loading website.json:", e)
#     exit(1)

# # OpenAI client
# client = OpenAI()

# # Load sentence transformer model for text similarity
# model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# def detect_language(query):
#     """Detects language of the query and returns language code."""
#     try:
#         lang = detect(query)
#         if lang not in ["en", "hi", "gu", "wa"]:  # Only support English, Hindi, Gujarati, WhatsApp-style language
#             return "en"  # Default to English
#         return lang
#     except:
#         return "en"

# def search_faiss(query):
#     """Searches FAISS and retrieves the most relevant product details."""
#     query_vector = model.encode(query).astype("float32")
#     _, indices = index.search(np.array([query_vector]), k=3)
#     results = [df_products.iloc[i].to_dict() for i in indices[0] if i != -1]
    
#     # Rank results based on text similarity
#     best_match = max(results, key=lambda x: util.pytorch_cos_sim(query_vector, model.encode(x["description"])).item(), default=None)
    
#     return json.dumps(best_match, ensure_ascii=False) if best_match else ""

# def get_website_info(category):
#     """Fetch relevant website details for contact, policy, or allergy queries."""
#     try:
#         if category == "contact":
#             contact = website_data.get("company", {}).get("contact", {})
#             return [
#                 f"Email: {contact.get('email', 'Not available')}",
#                 f"Address: {contact.get('address', 'Not available')}"
#             ]
#         elif category == "policy":
#             return [
#                 website_data.get("company", {}).get("shipping", {}).get("policy", "Shipping policy not available."),
#                 website_data.get("company", {}).get("refund_policy", {}).get("summary", "Refund policy not available.")
#             ]
#         elif category == "allergy":
#             return [
#                 "Check ingredients for allergens.",
#                 "No allergy guarantee.",
#                 "Consult doctor if unsure."
#             ]
#     except Exception as e:
#         print(f"Error retrieving website info: {e}")
#     return []

# def generate_replies(query, context, lang):
#     """Generates smart reply suggestions based on context and language."""
#     try:
#         # Check for restricted queries
#         restricted_queries = {"email": "contact", "contact number": "contact", "phone number": "contact", "address": "contact", "policy": "policy", "refund": "policy", "shipping": "policy", "allergy": "allergy", "allergic": "allergy"}
#         for word, category in restricted_queries.items():
#             if word in query.lower():
#                 return get_website_info(category)
        
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": f"You are a professional e-commerce assistant. Always reply in {lang}. Keep responses concise and professional, ensuring full sentences and no cutoff."},
#                 {"role": "user", "content": f"Customer Query: {query}\n\nRelevant Info: {context}\n\nProvide exactly three concise and professional reply suggestions in {lang}. Each response should be under 30 words and must be complete."}
#             ],
#             temperature=0.3,
#             max_tokens=50,  # Even tighter for brevity
#             n=1
#         )

#         if response and response.choices:
#             replies = response.choices[0].message.content.split("\n")
#             replies = [r.strip("-•1234567890. ") for r in replies if r.strip()]
#         else:
#             replies = ["Sorry, I couldn't generate a response."]
#     except Exception as e:
#         print(f"Error in generating reply: {e}")
#         replies = ["I'm sorry, something went wrong. Please try again."]
    
#     return replies[:3]

# def main():
#     print("FAISS index loaded successfully.")
#     print("Website info loaded successfully.\n")
    
#     while True:
#         query = input("Enter customer query (or type 'exit' to quit): ")
#         if query.lower() == 'exit':
#             break
        
#         lang = detect_language(query)
#         context = search_faiss(query)
#         replies = generate_replies(query, context, lang)
        
#         print("\nSmart Reply Suggestions:")
#         for i, reply in enumerate(replies, 1):
#             print(f"{i}. {reply}")
#         print()

# if __name__ == "__main__":
#     main()
