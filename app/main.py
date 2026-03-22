from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from app.services.models import ChatRequest, ConfigUpdate, ChatResponse, SecurityDetail, LogEntry
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

# --- THE MLOPS STATE ---
GLOBAL_THRESHOLD = 0.45
GLOBAL_MODEL = "LR_models" 

@app.on_event("startup")
async def startup_event():
    global GLOBAL_THRESHOLD, GLOBAL_MODEL
    setting = await settings_collection.find_one({"_id": "config"})
    
    if setting:
        GLOBAL_THRESHOLD = setting.get("threshold", 0.45)
        GLOBAL_MODEL = setting.get("model_folder", "LR_models")
    else:
        await settings_collection.insert_one({"_id": "config", "threshold": 0.45, "model_folder": "LR_models"})
    
    print(f"⚙️ Boot Configuration -> Threshold: {GLOBAL_THRESHOLD} | Model: {GLOBAL_MODEL}")
    scanner.load_model_from_folder(GLOBAL_MODEL)


# --- 1. SETTINGS ENDPOINTS ---
@app.get("/admin/settings/config")
async def get_config():
    return {"threshold": GLOBAL_THRESHOLD, "model_folder": GLOBAL_MODEL}

@app.post("/admin/settings/config")
async def update_config(update: ConfigUpdate):
    global GLOBAL_THRESHOLD, GLOBAL_MODEL
    
    GLOBAL_THRESHOLD = update.threshold
    
    if GLOBAL_MODEL != update.model_folder:
        GLOBAL_MODEL = update.model_folder
        success = scanner.load_model_from_folder(GLOBAL_MODEL)
        if not success:
            return {"error": f"Failed to load models from {GLOBAL_MODEL}"}

    await settings_collection.update_one(
        {"_id": "config"}, 
        {"$set": {"threshold": GLOBAL_THRESHOLD, "model_folder": GLOBAL_MODEL}}, 
        upsert=True
    )
    return {"message": "Configuration applied instantly.", "config": update.dict()}

# --- 2. CHAT ENDPOINTS ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    start_time = datetime.now(timezone.utc)
    
    is_safe, risk_score, triggers, layer_used, latency_ms = scanner.scan(request.message, threshold=GLOBAL_THRESHOLD)
    
    security_detail = SecurityDetail(
        scanner_name=layer_used, 
        is_safe=is_safe,
        risk_score=risk_score,
        triggers=triggers
    )

    if is_safe:
        status = "success"
        bot_reply = get_gemini_response(request.message) 
    else:
        status = "blocked"
        bot_reply = "⚠️ Security Alert: Your prompt was blocked by the SecureLLM Firewall due to potential injection patterns."

    log_dict = LogEntry(
        session_id=request.session_id,
        user_input=request.message,
        bot_response=bot_reply,
        is_safe=is_safe,
        risk_score=risk_score,
        triggers=triggers,
        timestamp=start_time
    ).dict()
    
    log_dict["layer_used"] = layer_used
    log_dict["latency_ms"] = latency_ms
    
    await chat_collection.insert_one(log_dict)

    bot_message_entry = {
        "role": "bot", 
        "content": bot_reply, 
        "timestamp": datetime.now(timezone.utc),
        "is_blocked": not is_safe
    }
    
    if not is_safe:
        sec_log_dict = security_detail.dict()
        sec_log_dict["latency_ms"] = latency_ms 
        bot_message_entry["security_log"] = sec_log_dict

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
    
    return {
        "status": status,
        "bot_reply": bot_reply,
        "security_log": {
            "scanner_name": layer_used,
            "is_safe": is_safe,
            "risk_score": risk_score,
            "triggers": triggers,
            "latency_ms": latency_ms
        },
        "timestamp": start_time,
        "session_id": request.session_id
    }

# ⚠️ FIXED HIERARCHY: Static Route placed BEFORE the Dynamic Route! 
@app.get("/chat/sessions")
async def list_all_sessions():
    """Returns a list of all chat sessions for the sidebar (Hyper-Safe Version)."""
    try:
        # Fetch the most recent 20 sessions without risky $slice operators
        cursor = sessions_collection.find({}, {"session_id": 1, "messages": 1}).sort("_id", -1).limit(20)
        sessions = await cursor.to_list(length=20)
        
        result = []
        for s in sessions:
            title = "Empty Chat"
            # Safely check if messages exist and is actually a list
            if "messages" in s and isinstance(s["messages"], list) and len(s["messages"]) > 0:
                title = str(s["messages"][0].get("content", "New Chat"))[:30] + "..."
                
            # Only append valid sessions
            if s.get("session_id"):
                result.append({
                    "session_id": s.get("session_id"),
                    "title": title
                })
                
        return {"sessions": result}
    
    except Exception as e:
        print(f"❌ DATABASE ERROR in list_all_sessions: {e}")
        # Return an empty list so the frontend doesn't crash!
        return {"sessions": []}

@app.get("/chat/sessions/{session_id}")
async def get_session_history(session_id: str):
    """Frontend calls this on reload to get the chat history back."""
    session = await sessions_collection.find_one({"session_id": session_id})
    if not session:
        return {"messages": []}
    messages = session.get("messages", [])
    
    for msg in messages:
        if "timestamp" in msg and msg["timestamp"].tzinfo is None:
            msg["timestamp"] = msg["timestamp"].replace(tzinfo=timezone.utc)
            
    return {"messages": messages}

# --- 3. ADMIN DASHBOARD ENDPOINTS ---
@app.get("/admin/stats")
async def get_dashboard_stats():
    total_requests = await chat_collection.count_documents({})
    blocked_requests = await chat_collection.count_documents({"is_safe": False})
    
    injection_rate = 0.0
    if total_requests > 0:
        injection_rate = round((blocked_requests / total_requests) * 100, 1)

    pipeline = [
        {"$match": {"is_safe": False}},          
        {"$unwind": "$triggers"},                
        {"$group": {"_id": "$triggers", "count": {"$sum": 1}}}, 
        {"$sort": {"count": -1}},                
        {"$limit": 4}                            
    ]
    top_triggers_cursor = chat_collection.aggregate(pipeline)
    top_patterns = [{"trigger": doc["_id"], "count": doc["count"]} async for doc in top_triggers_cursor]

    recent_logs = await chat_collection.find().sort("timestamp", -1).limit(10).to_list(10)
    for log in recent_logs:
        log["_id"] = str(log["_id"])
        if "timestamp" in log and log["timestamp"].tzinfo is None:
            log["timestamp"] = log["timestamp"].replace(tzinfo=timezone.utc)

    return {
        "total_requests": total_requests,
        "blocked_count": blocked_requests,
        "safe_count": total_requests - blocked_requests,
        "injection_rate": f"{injection_rate}%",
        "top_patterns": top_patterns,
        "recent_logs": recent_logs
    }