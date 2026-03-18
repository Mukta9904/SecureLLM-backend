import os
import motor.motor_asyncio
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGODB_URL")

# Create Client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client.secure_rag_db  # This will create a DB named 'secure_rag_db'
chat_collection = db.chat_logs