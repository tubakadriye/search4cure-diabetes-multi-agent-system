from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
import os

# === Load credentials securely ===
load_dotenv()  # loads .env from project root by default

MONGO_USER = quote_plus(os.getenv("MONGO_USER"))
MONGO_PASSWORD = quote_plus(os.getenv("MONGO_PASSWORD"))
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
MONGO_DB = os.getenv("MONGO_DB")
APP_NAME = os.getenv("APP_NAME", "MyApp")

# === Build the connection URI ===
MONGODB_URI = (
    f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_CLUSTER}.mongodb.net/"
    f"{MONGO_DB}?retryWrites=true&w=majority&appName={APP_NAME}"
)
#uri = 'mongodb+srv://' + username + ':' + password + "@diamind.q4fmjuw.mongodb.net/?retryWrites=true&w=majority&appName=DiaMind"

# === Create a MongoDB client ===
try:
    mongodb_client = MongoClient(
        MONGODB_URI,
        server_api=ServerApi('1')
    )
    mongodb_client.admin.command("ping")
    print("✅ MongoDB connection established.")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB: {e}")
    mongodb_client = None
