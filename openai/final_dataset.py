# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Dict
# import json
# import os
# from pymongo import MongoClient
# from bson import ObjectId
# import uvicorn
# import requests

# # FastAPI app
# app = FastAPI()

# # MongoDB setup
# client = MongoClient("mongodb+srv://kodemindstech:99aUqQeG3Z685yGr@koregrowwhatsappmarketi.jfuydzg.mongodb.net/KoreGrowWhatsAppMarketingApp?retryWrites=true&w=majority")
# db = client["KoreGrowWhatsAppMarketingApp"]
# collection = db["dataset"]

# # Pydantic model
# class ShopifyConfig(BaseModel):
#     accessToken: str
#     shopDomain: str
#     apiVersion: str
#     client_id: str
#     shopid: str

# # Helper to get Shopify base URL
# def get_base_url(shop_domain: str, api_version: str):
#     return f"https://{shop_domain}/admin/api/{api_version}"

# # Get store contact info (including phone)
# def get_shopify_store_info(base_url, headers):
#     url = f"{base_url}/shop.json"
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         shop = response.json().get("shop", {})
#         return {
#             "email": shop.get("email", "support@example.com"),
#             "phone": shop.get("phone", "N/A"),
#             "address": f"{shop.get('address1', 'Address')}, {shop.get('city', 'City')}, {shop.get('province', 'State')}, {shop.get('country', 'Country')}, {shop.get('zip', '000000')}"
#         }
#     return {"email": "support@example.com", "phone": "N/A", "address": "Address, City, State, 000000, Country"}

# # Get shop policies
# def get_shopify_policies(base_url, headers):
#     url = f"{base_url}/policies.json"
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         return {
#             policy["title"].lower().replace(" ", "_"): policy["body"]
#             for policy in response.json().get("policies", [])
#         }
#     return {}

# # Main route to generate and store dataset
# @app.post("/generate-store-dataset")
# def generate_and_store_dataset(config: ShopifyConfig):
#     try:
#         base_url = get_base_url(config.shopDomain, config.apiVersion)
#         headers = {"X-Shopify-Access-Token": config.accessToken}

#         # Fetch data
#         policies = get_shopify_policies(base_url, headers)
#         contact_info = get_shopify_store_info(base_url, headers)

#         # Create final dataset
#         data = {
#             "policies": policies,
#             "contact_information": contact_info,
#             "faqs": [
#                 {"question": "What payment methods do you accept?", "answer": "Credit/debit cards, UPI, net banking, and COD."},
#                 {"question": "How do I track my order?", "answer": "Tracking link sent via email after shipping."},
#                 {"question": "Do you ship internationally?", "answer": "Shipping availability depends on store policy."},
#                 {"question": "Can I cancel my order?", "answer": "Check the cancellation window in the return policy."},
#                 {"question": "Is COD available?", "answer": "Subject to availability at checkout."},
#                 {"question": "What if my product is damaged?", "answer": "Contact support with images within 48 hours."},
#                 {"question": "How to reach support?", "answer": f"Email {contact_info['email']} or call {contact_info['phone']}."}
#             ]
#         }

#         # Save as file
#         os.makedirs("datasets", exist_ok=True)
#         filename = f"{config.shopDomain.replace('.', '_')}.json"
#         file_path = os.path.join("datasets", filename)

#         with open(file_path, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=4, ensure_ascii=False)

#         # Save to MongoDB
#         entry = {
#             "client_id": ObjectId(config.client_id),
#             "shopid": ObjectId(config.shopid),
#             "file_data": data
#         }
#         result = collection.insert_one(entry)

#         return {
#             "message": "Dataset generated and stored successfully.",
#             "file": filename,
#             "inserted_id": str(result.inserted_id)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# def root():
#     return {"message": "FastAPI is running!"}

# if __name__ == "__main__":
#     uvicorn.run("final_dataset:app", host="0.0.0.0", port=8000, reload=True)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import json
import os
from pymongo import MongoClient
from bson import ObjectId
import uvicorn
import requests

# FastAPI app
app = FastAPI()

# MongoDB setup
client = MongoClient("mongodb+srv://kodemindstech:99aUqQeG3Z685yGr@koregrowwhatsappmarketi.jfuydzg.mongodb.net/KoreGrowWhatsAppMarketingApp?retryWrites=true&w=majority")
db = client["KoreGrowWhatsAppMarketingApp"]
collection = db["dataset"]

# Input config model
class ShopifyConfig(BaseModel):
    accessToken: str
    shopDomain: str
    apiVersion: str
    client_id: str
    shopid: str

# Output contact info model
class StoreContactInfo(BaseModel):
    email: str = "support@example.com"
    phone: str = "N/A"
    address: str = "N/A"

# Helper to get Shopify base URL
def get_base_url(shop_domain: str, api_version: str):
    return f"https://{shop_domain}/admin/api/{api_version}"

# Get store contact info (including phone & address)
def get_shopify_store_info(base_url, headers) -> Dict:
    try:
        url = f"{base_url}/shop.json"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        shop = response.json().get("shop", {})

        # Compose address with fallbacks
        address = ", ".join(filter(None, [
            shop.get("address1"),
            shop.get("city"),
            shop.get("province"),
            shop.get("country"),
            shop.get("zip")
        ])) or "N/A"

        return StoreContactInfo(
            email=shop.get("email", "support@example.com"),
            phone=shop.get("phone", "N/A"),
            address=address
        ).dict()

    except Exception:
        return StoreContactInfo().dict()

# Get shop policies
def get_shopify_policies(base_url, headers) -> Dict:
    try:
        url = f"{base_url}/policies.json"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return {
            policy["title"].lower().replace(" ", "_"): policy["body"]
            for policy in response.json().get("policies", [])
        }
    except Exception:
        return {}

# Main route to generate and store dataset
@app.post("/generate-store-dataset")
def generate_and_store_dataset(config: ShopifyConfig):
    try:
        base_url = get_base_url(config.shopDomain, config.apiVersion)
        headers = {"X-Shopify-Access-Token": config.accessToken}

        # Fetch data
        policies = get_shopify_policies(base_url, headers)
        contact_info = get_shopify_store_info(base_url, headers)

        # Create final dataset
        data = {
            "policies": policies,
            "contact_information": contact_info,
            "faqs": [
                {"question": "What payment methods do you accept?", "answer": "Credit/debit cards, UPI, net banking, and COD."},
                {"question": "How do I track my order?", "answer": "Tracking link sent via email after shipping."},
                {"question": "Do you ship internationally?", "answer": "Shipping availability depends on store policy."},
                {"question": "Can I cancel my order?", "answer": "Check the cancellation window in the return policy."},
                {"question": "Is COD available?", "answer": "Subject to availability at checkout."},
                {"question": "What if my product is damaged?", "answer": "Contact support with images within 48 hours."},
                {"question": "How to reach support?", "answer": f"Email {contact_info['email']} or call {contact_info['phone']}."}
            ]
        }

        # Save as file
        os.makedirs("datasets", exist_ok=True)
        filename = f"{config.shopDomain.replace('.', '_')}.json"
        file_path = os.path.join("datasets", filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # Save to MongoDB
        entry = {
            "client_id": ObjectId(config.client_id),
            "shopid": ObjectId(config.shopid),
            "file_data": data
        }
        result = collection.insert_one(entry)

        return {
            "message": "Dataset generated and stored successfully.",
            "file": filename,
            "inserted_id": str(result.inserted_id)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "FastAPI is running!"}

if __name__ == "__main__":
    uvicorn.run("final_dataset:app", host="0.0.0.0", port=8000, reload=True)
