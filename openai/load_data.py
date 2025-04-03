import json
import pandas as pd

# Load Product Data
with open("data\products.json", "r", encoding="utf-8") as file:
    product_data = json.load(file)

# Load Other Information (e.g., policies, FAQs, contact info)
with open("data\website.json", "r", encoding="utf-8") as file:
    other_data = json.load(file)

# Convert JSON to DataFrame
df_product = pd.DataFrame(product_data)
df_other = pd.DataFrame(other_data)

# Merge both datasets
df_combined = pd.concat([df_product, df_other], ignore_index=True)

# Display the first few rows
print(df_combined.head())

# Save as CSV for backup
df_combined.to_csv("combined_data.csv", index=False)

import pandas as pd

# Load the combined dataset
df = pd.read_csv("combined_data.csv")  # Ensure the correct filename

# Display the first 3 and last 3 entries
print(df.head(3))
print(df.tail(3))


# # Load the dataset
# df = pd.read_csv("combined_data.csv")

# Drop completely NaN rows
df = df.dropna(subset=["description"])  # Keep only rows with valid descriptions

# Convert non-string entries in 'company' to empty strings
df["company"] = df["company"].astype(str)

# Print first and last 3 entries again to confirm cleaning
print(df.head(3))
print(df.tail(3))

# Save the cleaned dataset
df.to_csv("cleaned_data.csv", index=False)
