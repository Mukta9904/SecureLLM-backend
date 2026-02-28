from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from app.models import ChatRequest, ChatResponse, SecurityDetail, LogEntry
from app.security.scanner import SecureScanner
from app.services.gemini import get_gemini_response
from app.database import chat_collection

app = FastAPI(title="SecureLLM System")

# CORS (Allow Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scanner = SecureScanner()

# --- 1. CHAT ENDPOINT (With DB Logging) ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = datetime.utcnow()
    
    # A. Security Scan
    is_safe, risk_score, triggers = scanner.scan(request.message)
    
    security_detail = SecurityDetail(
        scanner_name="SecureLLM-Sparse-Ngram",
        is_safe=is_safe,
        risk_score=risk_score,
        triggers=triggers
    )

    bot_reply = None
    status = "blocked"

    # B. Decision Logic
    if is_safe:
        status = "success"
        bot_reply = get_gemini_response(request.message)
    else:
        # THE "PRO" DECISION: Informative Blocking
        status = "blocked"
        bot_reply = "⚠️ Security Alert: Your prompt was blocked by the SecureLLM Firewall due to potential injection patterns."

    # C. Async DB Logging (Non-blocking)
    log_entry = LogEntry(
        user_input=request.message,
        bot_response=bot_reply,
        is_safe=is_safe,
        risk_score=risk_score,
        triggers=triggers,
        timestamp=start_time
    )
    await chat_collection.insert_one(log_entry.dict())

    return ChatResponse(
        status=status,
        bot_reply=bot_reply,
        security_log=security_detail,
        timestamp=start_time
    )

# --- 2. ADMIN DASHBOARD ENDPOINTS ---

@app.get("/admin/stats")
async def get_dashboard_stats():
    """Returns stats for the dashboard charts"""
    total_requests = await chat_collection.count_documents({})
    blocked_requests = await chat_collection.count_documents({"is_safe": False})
    
    # Get recent logs (limit 10)
    recent_logs = await chat_collection.find().sort("timestamp", -1).limit(10).to_list(10)
    
    # Serialize ObjectId for JSON
    for log in recent_logs:
        log["_id"] = str(log["_id"])

    return {
        "total_requests": total_requests,
        "blocked_count": blocked_requests,
        "safe_count": total_requests - blocked_requests,
        "recent_logs": recent_logs
    }