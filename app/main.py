from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from app.services.models import ChatRequest, ChatResponse, SecurityDetail, LogEntry, ThresholdUpdate
from app.security.scanner import SecureScanner
from app.services.gemini import get_gemini_response # Assuming you have this
from app.services.database import chat_collection, sessions_collection, settings_collection

app = FastAPI(title="SecureLLM System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scanner = SecureScanner()

# --- THE ZERO-LATENCY THRESHOLD TRICK ---
GLOBAL_THRESHOLD = 0.30

@app.on_event("startup")
async def startup_event():
    """Load the threshold from DB into RAM when server starts."""
    global GLOBAL_THRESHOLD
    setting = await settings_collection.find_one({"_id": "config"})
    if setting and "threshold" in setting:
        GLOBAL_THRESHOLD = setting["threshold"]
    else:
        # Create it if it doesn't exist
        await settings_collection.insert_one({"_id": "config", "threshold": 0.30})
        GLOBAL_THRESHOLD = 0.30
    print(f"⚙️ Active Security Threshold set to: {GLOBAL_THRESHOLD}")

# --- 1. SETTINGS ENDPOINTS ---
@app.get("/admin/settings/threshold")
async def get_threshold():
    return {"threshold": GLOBAL_THRESHOLD}

@app.post("/admin/settings/threshold")
async def update_threshold(update: ThresholdUpdate):
    """Admin updates the threshold. Updates RAM and DB instantly."""
    global GLOBAL_THRESHOLD
    GLOBAL_THRESHOLD = update.threshold
    await settings_collection.update_one(
        {"_id": "config"}, 
        {"$set": {"threshold": GLOBAL_THRESHOLD}}, 
        upsert=True
    )
    return {"message": "Threshold updated successfully", "new_threshold": GLOBAL_THRESHOLD}

# --- 2. CHAT ENDPOINTS ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = datetime.utcnow()
    
    # Pass the ultra-fast global threshold to the scanner
    is_safe, risk_score, triggers = scanner.scan(request.message, threshold=GLOBAL_THRESHOLD)
    
    security_detail = SecurityDetail(
        scanner_name="SecureLLM-Sparse-Ngram",
        is_safe=is_safe,
        risk_score=risk_score,
        triggers=triggers
    )

    if is_safe:
        status = "success"
        bot_reply = get_gemini_response(request.message) # Pass history here if your gemini function supports it
    else:
        status = "blocked"
        bot_reply = "⚠️ Security Alert: Your prompt was blocked by the SecureLLM Firewall due to potential injection patterns."

    # A. Save the specific Security Log
    log_entry = LogEntry(
        session_id=request.session_id,
        user_input=request.message,
        bot_response=bot_reply,
        is_safe=is_safe,
        risk_score=risk_score,
        triggers=triggers,
        timestamp=start_time
    )
    await chat_collection.insert_one(log_entry.dict())

    # B. Update the Chat Session History
    bot_message_entry = {
        "role": "bot", 
        "content": bot_reply, 
        "timestamp": datetime.utcnow(),
        "is_blocked": not is_safe
    }
    
    if not is_safe:
        # Attach the forensic data to the history so the UI can reconstruct the red box
        bot_message_entry["security_log"] = security_detail.dict()

    await sessions_collection.update_one(
        {"session_id": request.session_id},
        {"$push": {
            "messages": {"$each": [
                {"role": "user", "content": request.message, "timestamp": start_time},
                bot_message_entry
            ]}
        }},
        upsert=True
    )
    return ChatResponse(
        status=status,
        bot_reply=bot_reply,
        security_log=security_detail,
        timestamp=start_time,
        session_id=request.session_id
    )

@app.get("/chat/sessions/{session_id}")
async def get_session_history(session_id: str):
    """Frontend calls this on reload to get the chat history back."""
    session = await sessions_collection.find_one({"session_id": session_id})
    if not session:
        return {"messages": []}
    return {"messages": session.get("messages", [])}

# --- 3. ADMIN DASHBOARD ENDPOINTS ---
@app.get("/admin/stats")
async def get_dashboard_stats():
    """Returns REAL stats for the dashboard charts"""
    total_requests = await chat_collection.count_documents({})
    blocked_requests = await chat_collection.count_documents({"is_safe": False})
    
    # Calculate real injection rate
    injection_rate = 0.0
    if total_requests > 0:
        injection_rate = round((blocked_requests / total_requests) * 100, 1)

    # MongoDB Magic: Find the most common trigger words automatically
    pipeline = [
        {"$match": {"is_safe": False}},          # Only look at blocked prompts
        {"$unwind": "$triggers"},                # Unpack the list of triggers
        {"$group": {"_id": "$triggers", "count": {"$sum": 1}}}, # Count them
        {"$sort": {"count": -1}},                # Sort highest to lowest
        {"$limit": 4}                            # Get top 4
    ]
    top_triggers_cursor = chat_collection.aggregate(pipeline)
    top_patterns = [{"trigger": doc["_id"], "count": doc["count"]} async for doc in top_triggers_cursor]

    # Get recent logs
    recent_logs = await chat_collection.find().sort("timestamp", -1).limit(10).to_list(10)
    for log in recent_logs:
        log["_id"] = str(log["_id"])

    return {
        "total_requests": total_requests,
        "blocked_count": blocked_requests,
        "safe_count": total_requests - blocked_requests,
        "injection_rate": f"{injection_rate}%",
        "top_patterns": top_patterns,
        "recent_logs": recent_logs
    }
    
@app.get("/chat/sessions")
async def list_all_sessions():
    """Returns a list of all chat sessions for the sidebar."""
    # Grab the most recent 20 sessions, but only fetch the very first message of each to use as the title
    cursor = sessions_collection.find({}, {"session_id": 1, "messages": {"$slice": 1}}).sort("_id", -1).limit(20)
    sessions = await cursor.to_list(length=20)
    
    result = []
    for s in sessions:
        title = "Empty Chat"
        if "messages" in s and len(s["messages"]) > 0:
            # Use the first 30 characters of the first message as the title
            title = s["messages"][0].get("content", "New Chat")[:30] + "..."
            
        result.append({
            "session_id": s.get("session_id"),
            "title": title
        })
        
    return {"sessions": result}