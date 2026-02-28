from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# --- API Models (Frontend <-> Backend) ---
class ChatRequest(BaseModel):
    message: str

class SecurityDetail(BaseModel):
    scanner_name: str
    is_safe: bool
    risk_score: float
    triggers: List[str]

class ChatResponse(BaseModel):
    status: str
    bot_reply: Optional[str] = None
    security_log: SecurityDetail
    timestamp: datetime

# --- Database Model (Backend <-> MongoDB) ---
class LogEntry(BaseModel):
    user_input: str
    bot_response: Optional[str]
    is_safe: bool
    risk_score: float
    triggers: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)