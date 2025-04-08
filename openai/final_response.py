from flask import Flask, request, jsonify
import json
import openai
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API Key not found in .env file.")

# MongoDB connection
#mongo_uri = os.getenv("MONGODB_URI") or "your_mongodb_uri"
mongo_uri = "mongodb+srv://kodemindstech:99aUqQeG3Z685yGr@koregrowwhatsappmarketi.jfuydzg.mongodb.net/KoreGrowWhatsAppMarketingApp?retryWrites=true&w=majority"
mongo_client = MongoClient(mongo_uri)
db = mongo_client["KoreGrowWhatsAppMarketingApp"]
collection = db["dataset"]

app = Flask(__name__)

# Store short-term context per client_id
conversation_history = {}

def fetch_dataset_from_mongo(client_id):
    """Fetch dataset from MongoDB using client_id."""
    try:
        entry = collection.find_one({"client_id":  ObjectId(client_id)})
        if not entry or "file_data" not in entry:
            raise ValueError("Dataset not found for this client ID.")
        return entry["file_data"]
    except Exception as e:
        raise ValueError(f"Failed to fetch dataset: {str(e)}")

def find_product_suggestion(query, client_data):
    keywords = client_data.get("keywords", {})
    for key, category in keywords.items():
        if key in query.lower():
            products = client_data.get("product_categories", {}).get(category, [])
            if products:
                return products[0]
    return None

def generate_ai_response(client_id, query, client_data, product=None):
    if client_id not in conversation_history:
        conversation_history[client_id] = []

    context = "\n".join(conversation_history[client_id][-2:])
    client_name = client_data.get("client_name", "E-commerce Assistant")

    prompt = f"""
    You are a smart e-commerce assistant for {client_name}.
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
    data = request.json
    query = data.get("message", "")
    client_id = data.get("client_id", "")

    if not query or not client_id:
        return jsonify({"error": "Message and client_id are required"}), 400

    try:
        client_data = fetch_dataset_from_mongo(client_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if client_id not in conversation_history:
        conversation_history[client_id] = []
    conversation_history[client_id].append(f"User: {query}")

    product = find_product_suggestion(query, client_data)
    ai_response = generate_ai_response(client_id, query, client_data, product)

    conversation_history[client_id].append(f"Assistant: {ai_response}")

    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6000)
