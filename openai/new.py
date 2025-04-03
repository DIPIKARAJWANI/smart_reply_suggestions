# import json
# import openai
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env
# load_dotenv()

# # Get OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# if not openai.api_key:
#     print("⚠️ Error: OpenAI API Key not found in .env file.")
#     exit()

# # Load data.json
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(BASE_DIR, "data\data.json")

# try:
#     with open(DATA_PATH, "r", encoding="utf-8") as file:
#         data = json.load(file)
# except FileNotFoundError:
#     print("⚠️ Error: 'data.json' not found.")
#     exit()

# # Store past messages for short-term context (2-3 messages)
# conversation_history = []

# def find_product_suggestion(query):
#     """Find a relevant product based on user query."""
#     keywords = {
#         "weight gain": "weight_gainers",
#         "gain weight": "weight_gainers",
#         "wajan badhana": "weight_gainers",
#         "weight loss": "fat_burners",
#         "lose weight": "fat_burners",
#         "wajan kam karna": "fat_burners",
#         "whey protein": "whey_proteins",
#         "pre-workout": "pre_post_workout",
#         "fat burner": "fat_burners",
#         "multivitamin": "vitamins_minerals"
#     }

#     for key, category in keywords.items():
#         if key in query.lower():
#             products = data["product_categories"].get(category, [])
#             if products:
#                 return products[0]  

#     return None

# def generate_ai_response(query, product=None):
#     """Generate a very short, precise, and website-specific reply using OpenAI."""

#     context = "\n".join(conversation_history[-2:])  

#     prompt = f"""
#     You are a smart e-commerce assistant for 2X Nutrition.
#     - Replies must be **VERY SHORT (1-2 lines), complete, and website-related**.
#     - No general fitness advice—only reply using product details, policies, or FAQs.
#     - Use the same language as the customer query.
#     - If a product is relevant, mention it briefly.

#     Context:
#     {context}

#     User: {query}
#     """

#     if product:
#         prompt += f"\nProduct Info: {product['name']} ({product['price']}) - {product['description']}."

#     prompt += "\nAssistant (short and website-related reply):"

#     response = openai.ChatCompletion.create(
#         model="gpt-4-turbo",
#         messages=[{"role": "system", "content": prompt}],
#         max_tokens=40  
#     )

#     return response["choices"][0]["message"]["content"].strip()

# def main():
#     print("\nSmart Reply System Initialized!\n")

#     while True:
#         query = input("Enter customer query (or type 'exit' to quit): ")
#         if query.lower() == "exit":
#             break

#         conversation_history.append(f"User: {query}")

#         product = find_product_suggestion(query)
#         ai_response = generate_ai_response(query, product)

#         conversation_history.append(f"Assistant: {ai_response}")

#         print("\nSmart Reply Suggestions:")
#         print(f"1. {ai_response}\n")

# if __name__ == "__main__":
#     main()


from flask import Flask, request, jsonify
import json
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check API Key
if not openai.api_key:
    raise ValueError("⚠️ Error: OpenAI API Key not found in .env file.")

# Load product data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/data.json")

try:
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
except FileNotFoundError:
    raise FileNotFoundError("⚠️ Error: 'data.json' not found.")

app = Flask(__name__)

# Store past messages for short-term context (2-3 messages)
conversation_history = []

def find_product_suggestion(query):
    """Find a relevant product based on user query."""
    keywords = {
        "weight gain": "weight_gainers",
        "gain weight": "weight_gainers",
        "wajan badhana": "weight_gainers",
        "weight loss": "fat_burners",
        "lose weight": "fat_burners",
        "wajan kam karna": "fat_burners",
        "whey protein": "whey_proteins",
        "pre-workout": "pre_post_workout",
        "fat burner": "fat_burners",
        "multivitamin": "vitamins_minerals"
    }

    for key, category in keywords.items():
        if key in query.lower():
            products = data["product_categories"].get(category, [])
            if products:
                return products[0]  

    return None

def generate_ai_response(query, product=None):
    """Generate a very short AI response using OpenAI."""
    context = "\n".join(conversation_history[-2:])

    prompt = f"""
    You are a smart e-commerce assistant for 2X Nutrition.
    - Replies must be **VERY SHORT (1-2 lines), complete, and website-related**.
    - No general fitness advice—only reply using product details, policies, or FAQs.
    - Use the same language as the customer query.
    - If a product is relevant, mention it briefly.

    Context:
    {context}

    User: {query}
    """

    if product:
        prompt += f"\nProduct Info: {product['name']} ({product['price']}) - {product['description']}."

    prompt += "\nAssistant (short and website-related reply):"

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=30
    )

    return response["choices"][0]["message"]["content"].strip()

@app.route("/generateresponse", methods=["POST"])
def generate_response():
    """API endpoint to generate AI response."""
    data = request.json
    query = data.get("message", "")
    print("req message:",data)

    if not query:
        return jsonify({"error": "Message is required"}), 400

    conversation_history.append(f"User: {query}")
    product = find_product_suggestion(query)
    ai_response = generate_ai_response(query, product)

    conversation_history.append(f"Assistant: {ai_response}")

    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6000)
