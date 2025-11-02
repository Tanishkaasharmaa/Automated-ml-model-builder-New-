import pickle
import os
from typing import Any, Dict

SESSION_FILE = "sessions.pkl"
SESSIONS: Dict[str, Dict[str, Any]] = {}


# --- Load sessions when app starts ---
if os.path.exists(SESSION_FILE):
    try:
        with open(SESSION_FILE, "rb") as f:
            SESSIONS = pickle.load(f)
        print(f"[Session] Loaded {len(SESSIONS)} sessions from disk.")
    except Exception as e:
        print("[Session] Failed to load sessions:", e)


# --- Save sessions whenever updated ---
def save_sessions():
    try:
        with open(SESSION_FILE, "wb") as f:
            pickle.dump(SESSIONS, f)
    except Exception as e:
        print("[Session] Error saving sessions:", e)


def get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve or create a session by ID."""
    session = SESSIONS.setdefault(session_id, {})
    save_sessions()  # save immediately after creation
    return session


def set_session(session_id: str, data: Dict[str, Any]):
    """Store or update a session."""
    SESSIONS[session_id] = data
    save_sessions()


def clear_session(session_id: str):
    """Delete a session."""
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        save_sessions()
