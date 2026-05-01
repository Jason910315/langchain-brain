import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

"""
為了管理使用者的對話紀錄，對 Supabase 的 table 進行 CRUD 操作
"""

# _client 全域變數，可以是兩種型別 (所有函式都可以直接用不必傳入參數)
_client: Client | None = None

def _get_client() -> Client:
    global _client   # 初次建立後每次直接拿全域就好，因為他會一值保存

    # 初次進入還未建立 Supabase 連線，先建立
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL 或 SUPABASE_KEY 未設定，請於 .env 中設定")
        
        _client = create_client(url, key)
    return _client

def create_session(user_id: str, title: str) -> str:
    """
    使用者建立一個新的對話，回傳 session_id
    """
    client = _get_client()
    result = client.table("sessions").insert({
        "user_id": user_id,
        "title": title[:40], 
    }).execute()
    return result.data[0]["id"]  # id 是主鍵 UUID，會自動產生

def save_message(session_id: str, role: str, content: str, sources: list | None = None) -> None:
    """
    儲存 session_id 對話紀錄中的單則訊息
    """
    client = _get_client()
    client.table("messages").insert({
        "session_id": session_id,
        "role": role,   # 'user | assistant'
        "content": content,
        "sources": sources,
    }).execute()


def load_messages(session_id: str) -> list[dict]:
    """
    取出該 session_id 內所有歷史對話紀錄，為了:
        - 顯示歷史對話紀錄於前端
        - 給予 RAG 回答時，可以參考前面歷史對話的內容
    """
    client = _get_client()

    # 查詢該 session_id 內所有對話，並依照時間排序 (最早的在前)
    result = (
        client.table("messages")
        .select("role, content, sources")
        .eq("session_id", session_id)
        .order("created_at").execute()  # 預設是 asc，由早到晚排
    )
    return result.data

def list_sessions(user_id: str) -> list[dict]:
    """
    取出該使用者的所有對話清單
    """

    client = _get_client()
    result = (
        client.table("sessions")
        .select("id, title, created_at")
        .eq("user_id", user_id)   # 每位 user 有自己的多個 session 紀錄
        .order("created_at", desc=True).execute()
    )
    return result.data

def delete_session(session_id: str) -> None:
    """
    刪除整個對話紀錄，且 messages table 參考到該 session_id 的所有資料也會一併刪除 (CASECADE)
    """
    client = _get_client()
    client.table("sessions").delete().eq("id", session_id).execute()