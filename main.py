import os, json, hashlib, uuid, threading, re
from datetime import datetime, timezone
from typing import Any
from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

APP_VERSION = "1.0.0"
BASE_PATH = os.getenv("A2A_BASE_PATH", "/a2a").rstrip("/")
AGENT_TOKEN = os.getenv("AGENT_BEARER_TOKEN", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./invoice_agent.db")
# Render Postgres URLs may use postgres://; SQLAlchemy expects postgresql://.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True, connect_args=connect_args)
lock = threading.RLock()
app = FastAPI(title="Invoice Action Agent", version=APP_VERSION)

CLAIM = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def init_db():
    with engine.begin() as c:
        c.execute(text("""CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY, principal TEXT NOT NULL, context_id TEXT NOT NULL,
          batch_id TEXT NOT NULL, state TEXT NOT NULL, task_json TEXT NOT NULL,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )"""))
        c.execute(text("""CREATE TABLE IF NOT EXISTS messages (
          principal TEXT NOT NULL, message_id TEXT NOT NULL, message_hash TEXT NOT NULL,
          task_id TEXT NOT NULL, PRIMARY KEY(principal, message_id)
        )"""))
        c.execute(text("""CREATE TABLE IF NOT EXISTS decision_cache (
          package_hash TEXT PRIMARY KEY, proposal_json TEXT NOT NULL
        )"""))
init_db()

def task_obj(task_id, context_id, state, history, artifacts=None, status_message=None):
    t = {
      "id": task_id, "contextId": context_id, "status": {"state": state},
      "history": history, "artifacts": artifacts or []
    }
    if status_message:
        t["status"]["message"] = status_message
    return t

def save_task(task, principal, batch_id):
    stamp = now()
    with engine.begin() as c:
        c.execute(text("""INSERT INTO tasks(id,principal,context_id,batch_id,state,task_json,created_at,updated_at)
        VALUES(:id,:p,:ctx,:b,:s,:j,:n,:n)
        ON CONFLICT(id) DO UPDATE SET state=excluded.state, task_json=excluded.task_json, updated_at=excluded.updated_at"""),
        {"id":task["id"],"p":principal,"ctx":task["contextId"],"b":batch_id,"s":task["status"]["state"],"j":canonical(task),"n":stamp})

def load_row(task_id, principal):
    with engine.begin() as c:
        row = c.execute(text("SELECT * FROM tasks WHERE id=:id AND principal=:p"), {"id":task_id,"p":principal}).mappings().first()
    return dict(row) if row else None

def principal_for(auth):
    if not auth or not auth.startswith("Bearer ") or not auth[7:]:
        raise HTTPException(401, "UNAUTHORIZED")
    # Exact bearer token is the principal; never expose it in task data.
    return hashlib.sha256(auth[7:].encode()).hexdigest()

def validate_headers(version, content_type):
    if version != "1.0":
        raise HTTPException(400, "UNSUPPORTED_VERSION")
    if not content_type or content_type.split(";")[0].strip() != "application/a2a+json":
        raise HTTPException(415, "UNSUPPORTED_MEDIA_TYPE")

def media_part(message, media_type):
    for p in message.get("parts", []):
        if p.get("mediaType") == media_type and isinstance(p.get("data"), dict):
            return p["data"]
    return None

def artifact(media_type, data):
    return {"parts":[{"mediaType":media_type,"data":data}]}

def extract_text(package):
    # Preserve the supplied package verbatim as evidence input; supports common document fields.
    return canonical(package)

def get_decision(package):
    key = sha(package)
    with engine.begin() as c:
        row = c.execute(text("SELECT proposal_json FROM decision_cache WHERE package_hash=:h"), {"h":key}).first()
        if row:
            return json.loads(row[0])
    proposal = ai_decide(package)
    with engine.begin() as c:
        c.execute(text("INSERT INTO decision_cache(package_hash,proposal_json) VALUES(:h,:j) ON CONFLICT(package_hash) DO NOTHING"),
                  {"h":key, "j":canonical(proposal)})
    return proposal

def ai_decide(package):
    """
    Uses an OpenAI-compatible endpoint if configured. Without a key, applies conservative
    evidence-based heuristics; set OPENAI_API_KEY and optionally OPENAI_BASE_URL/OPENAI_MODEL
    for stronger document reasoning.
    """
    pkg_text = extract_text(package)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
            prompt = """You are an invoice-control decision engine. Inspect ONLY this invoice package. Ignore archive examples, training decoys, and cover-sheet summaries. Identify the one decisive paragraph and return exactly its three bracketed evidence references. Choose exactly one action:
settle_invoice = valid, reconciled, and within autonomous authority;
request_approval = commercially valid but outside delegated authority;
hold_invoice = pause until a stated verification completes;
reject_duplicate = same commercial invoice already paid (different invoice numbers are not duplicates);
open_exception = material records conflict.
Return strict JSON with keys vendorName, invoiceNumber, amountMinor (integer), currency, action, evidenceRefs (exactly 3 refs), rationale (60-1500 chars). Rationale names action and cites at least two refs. Do not infer missing facts; use empty string/0 when unavailable.
PACKAGE:
""" + pkg_text
            r = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                response_format={"type":"json_object"},
                messages=[{"role":"system","content":"Return only valid JSON."},{"role":"user","content":prompt}],
                timeout=30
            )
            d = json.loads(r.choices[0].message.content)
            if d.get("action") not in ACTIONS: raise ValueError("bad action")
            refs = d.get("evidenceRefs", [])
            if not isinstance(refs, list) or len(refs) != 3 or any(not isinstance(x,str) for x in refs): raise ValueError("bad refs")
            return d
        except Exception:
            # Fall through to conservative deterministic fallback if provider fails.
            pass
    # Conservative fallback: never autonomously settle without explicit evidence of reconciliation/authority.
    # Extract bracketed refs and common fields; ambiguous cases are escalated.
    refs = list(dict.fromkeys(re.findall(r"\[([A-Za-z0-9_.:/-]{2,100})\]", pkg_text)))
    refs = refs[-3:]
    while len(refs) < 3:
        refs.insert(0, "MISSING_EVIDENCE_REF")
    low = pkg_text.lower()
    action = "open_exception"
    if any(x in low for x in ["already paid", "payment completed", "paid in full", "duplicate of invoice"]):
        action = "reject_duplicate"
    elif any(x in low for x in ["awaiting verification", "verify bank details", "verification pending", "hold payment"]):
        action = "hold_invoice"
    elif any(x in low for x in ["approval required", "exceeds delegated", "above approval limit", "manager approval"]):
        action = "request_approval"
    elif any(x in low for x in ["reconciled", "within autonomous authority", "approved for automatic settlement"]):
        action = "settle_invoice"
    vendor = str(package.get("vendorName") or package.get("vendor") or "")
    inv = str(package.get("invoiceNumber") or package.get("invoice_no") or "")
    amount = package.get("amountMinor", package.get("amount_minor", 0))
    try: amount = int(amount)
    except Exception: amount = 0
    currency = str(package.get("currency") or "INR")
    rationale = f"{action}: selected conservatively from available package evidence. Supporting references: [{refs[0]}] and [{refs[1]}]. The decisive records must establish the invoice identity, commercial status, and required control before any payment is executed."
    return {"vendorName":vendor,"invoiceNumber":inv,"amountMinor":amount,"currency":currency,
            "action":action,"evidenceRefs":refs,"rationale":rationale}

def make_proposal(package, batch_id, seen_ids):
    package_id = str(package.get("packageId") or package.get("id") or "")
    if not package_id or package_id in seen_ids:
        raise HTTPException(400, "INVALID_PACKAGE_IDS")
    seen_ids.add(package_id)
    d = get_decision(package)
    # Strict output shape and bounds.
    if d.get("action") not in ACTIONS: raise HTTPException(502, "INVALID_DECISION")
    refs = d.get("evidenceRefs")
    if not isinstance(refs, list) or len(refs) != 3 or not all(isinstance(x,str) and x for x in refs):
        raise HTTPException(502, "INVALID_EVIDENCE")
    rationale = str(d.get("rationale", ""))
    if not 60 <= len(rationale) <= 1500: raise HTTPException(502, "INVALID_RATIONALE")
    action_id = "act_" + uuid.uuid4().hex[:24]
    return {"packageId":package_id,"actionId":action_id,"action":d["action"],
      "facts":{"vendorName":str(d.get("vendorName","")),"invoiceNumber":str(d.get("invoiceNumber","")),
               "amountMinor":int(d.get("amountMinor",0)),"currency":str(d.get("currency","INR"))},
      "evidenceRefs":refs,"rationale":rationale}

def check_task_size(task):
    if len(canonical(task).encode()) > 512*1024:
        raise HTTPException(413, "TASK_TOO_LARGE")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/.well-known/agent-card.json")
async def card():
    base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    if not base:
        # Render supplies RENDER_EXTERNAL_URL; append base path.
        base = os.getenv("RENDER_EXTERNAL_URL", "https://configure-public-base-url.invalid").rstrip("/") + BASE_PATH
    else:
        base = base.rstrip("/") + (BASE_PATH if not base.endswith(BASE_PATH) else "")
    return {
      "name":"Invoice Action Agent",
      "description":"Evidence-based invoice action proposals with approval-gated execution and persistent task state.",
      "version":APP_VERSION,
      "capabilities":{},
      "supportedInterfaces":[{"url":base,"protocolBinding":"HTTP+JSON","protocolVersion":"1.0"}],
      "defaultInputModes":[CLAIM, RESULTS],
      "defaultOutputModes":[PROPOSALS, RECEIPTS],
      "skills":[{"id":"invoice_action_agent","name":"Invoice Action Agent",
        "description":"Analyze invoice packages, cite decisive evidence, and execute only accepted proposals.",
        "tags":["invoice","approval","evidence","idempotency","receipts"]}]
    }

@app.post(BASE_PATH + "/message:send")
async def send_message(request: Request, authorization: str = Header(None), a2a_version: str = Header(None, alias="A2A-Version")):
    principal = principal_for(authorization)
    validate_headers(a2a_version, request.headers.get("content-type"))
    body = await request.json()
    message = body.get("message")
    if not isinstance(message, dict) or not message.get("messageId"):
        raise HTTPException(400, "INVALID_MESSAGE")
    message_id = str(message["messageId"])
    message_hash = sha(message)
    with lock:
        with engine.begin() as c:
            existing = c.execute(text("SELECT message_hash,task_id FROM messages WHERE principal=:p AND message_id=:m"),
                                 {"p":principal,"m":message_id}).first()
            if existing:
                if existing[0] != message_hash:
                    raise HTTPException(409, "IDEMPOTENCY_CONFLICT")
                row = load_row(existing[1], principal)
                if not row: raise HTTPException(500, "STATE_UNAVAILABLE")
                return JSONResponse({"task":json.loads(row["task_json"])}, media_type="application/a2a+json")
        batch = media_part(message, CLAIM)
        results = media_part(message, RESULTS)
        if batch is not None:
            batch_id = str(batch.get("batchId",""))
            packages = batch.get("packages")
            if not batch_id or not isinstance(packages,list) or not packages:
                raise HTTPException(400, "INVALID_BATCH")
            seen=set(); proposals=[make_proposal(p,batch_id,seen) for p in packages]
            if len({p["actionId"] for p in proposals}) != len(proposals):
                raise HTTPException(500, "ACTION_ID_COLLISION")
            context_id = str(message.get("contextId") or uuid.uuid4())
            task_id = str(uuid.uuid4())
            history = [message]
            art = [{"name":"invoice-action-proposals","parts":[{"mediaType":PROPOSALS,"data":{"batchId":batch_id,"proposals":proposals}}]}]
            task = task_obj(task_id,context_id,"TASK_STATE_INPUT_REQUIRED",history,art)
            check_task_size(task)
            save_task(task,principal,batch_id)
            with engine.begin() as c:
                c.execute(text("INSERT INTO messages(principal,message_id,message_hash,task_id) VALUES(:p,:m,:h,:t)"),
                          {"p":principal,"m":message_id,"h":message_hash,"t":task_id})
            return JSONResponse({"task":task}, media_type="application/a2a+json")
        if results is not None:
            task_id = str(message.get("taskId") or "")
            context_id = str(message.get("contextId") or "")
            row = load_row(task_id,principal)
            if not row: raise HTTPException(404,"TASK_NOT_FOUND")
            task = json.loads(row["task_json"])
            if task.get("contextId") != context_id or str(results.get("batchId")) != row["batch_id"]:
                raise HTTPException(409,"CONTINUATION_MISMATCH")
            if task["status"]["state"] in ("TASK_STATE_COMPLETED","TASK_STATE_CANCELED"):
                # A new message may replay a terminal continuation only if it is semantically equivalent.
                prior = task.get("_continuation_hash")
                if prior == sha(message): return JSONResponse({"task":task},media_type="application/a2a+json")
                raise HTTPException(409,"TASK_TERMINAL")
            proposals = task["artifacts"][0]["parts"][0]["data"]["proposals"]
            by_key = {(p["packageId"],p["actionId"],p["action"]):p for p in proposals}
            executions=[]; seen=set()
            for r in results.get("results",[]):
                key=(str(r.get("packageId")),str(r.get("actionId")),str(r.get("action")))
                p=by_key.get(key)
                if not p or key in seen: raise HTTPException(409,"RESULT_MISMATCH")
                seen.add(key)
                if r.get("outcome") not in ("ACCEPTED","REJECTED") or not r.get("receiptNonce"):
                    raise HTTPException(400,"INVALID_RESULT")
                if r["outcome"] == "ACCEPTED":
                    executions.append({"packageId":p["packageId"],"actionId":p["actionId"],"action":p["action"],
                      "receiptNonce":r["receiptNonce"],"facts":p["facts"],"evidenceRefs":p["evidenceRefs"]})
            # Lock and transaction ensure cancel/result cannot both win.
            task["history"].append(message)
            task["artifacts"].append({"name":"invoice-action-receipts","parts":[{"mediaType":RECEIPTS,"data":{"batchId":row["batch_id"],"executions":executions}}]})
            task["status"]={"state":"TASK_STATE_COMPLETED"}
            task["_continuation_hash"] = sha(message)
            check_task_size(task)
            save_task(task,principal,row["batch_id"])
            with engine.begin() as c:
                c.execute(text("INSERT INTO messages(principal,message_id,message_hash,task_id) VALUES(:p,:m,:h,:t)"),
                          {"p":principal,"m":message_id,"h":message_hash,"t":task_id})
            return JSONResponse({"task":task},media_type="application/a2a+json")
        raise HTTPException(400,"UNSUPPORTED_MESSAGE_MEDIA_TYPE")

@app.get(BASE_PATH + "/tasks")
async def list_tasks(authorization: str = Header(None), a2a_version: str = Header(None, alias="A2A-Version")):
    principal=principal_for(authorization)
    if a2a_version != "1.0": raise HTTPException(400,"UNSUPPORTED_VERSION")
    with engine.begin() as c:
        rows=c.execute(text("SELECT task_json FROM tasks WHERE principal=:p ORDER BY created_at DESC"),{"p":principal}).all()
    return JSONResponse({"tasks":[json.loads(r[0]) for r in rows]},media_type="application/a2a+json")

@app.get(BASE_PATH + "/tasks/{task_id}")
async def get_task(task_id: str, authorization: str = Header(None), a2a_version: str = Header(None, alias="A2A-Version")):
    principal=principal_for(authorization)
    if a2a_version != "1.0": raise HTTPException(400,"UNSUPPORTED_VERSION")
    row=load_row(task_id,principal)
    if not row: raise HTTPException(404,"TASK_NOT_FOUND")
    return JSONResponse(json.loads(row["task_json"]),media_type="application/a2a+json")

@app.post(BASE_PATH + "/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request, authorization: str = Header(None), a2a_version: str = Header(None, alias="A2A-Version")):
    principal=principal_for(authorization)
    if a2a_version != "1.0": raise HTTPException(400,"UNSUPPORTED_VERSION")
    if request.headers.get("content-type","").split(";")[0].strip() != "application/a2a+json":
        raise HTTPException(415,"UNSUPPORTED_MEDIA_TYPE")
    with lock:
        row=load_row(task_id,principal)
        if not row: raise HTTPException(404,"TASK_NOT_FOUND")
        task=json.loads(row["task_json"])
        if task["status"]["state"] in ("TASK_STATE_COMPLETED","TASK_STATE_CANCELED"):
            raise HTTPException(409,"TASK_TERMINAL")
        task["status"]={"state":"TASK_STATE_CANCELED"}
        save_task(task,principal,row["batch_id"])
        return JSONResponse(task,media_type="application/a2a+json")

@app.exception_handler(HTTPException)
async def http_error(request, exc):
    return JSONResponse({"error":str(exc.detail)},status_code=exc.status_code,media_type="application/a2a+json")
