import os, requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from llama_index.core.schema import Document

load_dotenv()

# HackMD API 基礎 URL，所有請求都是由這裡發出
HACKMD_API_BASE = "https://api.hackmd.io/v1"

HACKMD_API_TOKEN = os.getenv("HACKMD_API_TOKEN")


def fetch_all_notes() -> list[dict]:
    """
    取得帳號下所有筆記的 metadata 資訊 (還不包含內文)，並返回一個清單
    """
    url = f"{HACKMD_API_BASE}/notes"
    headers = {
        "Authorization": f"Bearer {HACKMD_API_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)
    # 會在 HTTP 狀態為 4xx 或 5xx 時自動拋出例外
    response.raise_for_status()

    notes = response.json()
    print(f"[HackMD] 共找到 {len(notes)} 篇筆記")
    return notes

def fetch_note_content(note_id: str) -> str:
    """
    取得單篇筆記的完整 Markdown 內文，這裡回傳的物件筆 fecth_all_notes() 還多一個 content 欄位
    """
    # 取得 note_id 這篇筆記的內文
    url = f"{HACKMD_API_BASE}/notes/{note_id}"
    headers = {
        "Authorization": f"Bearer {HACKMD_API_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)
    # 會在 HTTP 狀態為 4xx 或 5xx 時自動拋出例外
    response.raise_for_status()

    note = response.json()
    # 只取 content 欄位出來 (內文)
    note_content = note.get("content", "")
    return note_content

def _parse_hackmd_timestamp(ts_ms: int) -> str:
    """
    HackMD 的時間戳是以 Unix Timestamp 毫秒為單位，需要轉換為 ISO 8601 格式 (例如: 2026-05-11T00:00:00.000Z)
    因為後續 Qdrant 內的每個 chunk 的 payload 存的 ingested_at 欄位是 ISO 8601 格式，所以需要轉換才能做比對
    """
    # 先將 ts_ms 轉成秒，再轉成 datetime 物件，最後轉成 ISO 8601 格式
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.isoformat()

def load_notes_as_documents(note_ids: list[str] = None) -> list[Document]:
    """
    將筆記轉為 LlamaIndex Documents 物件 (每一份資料，還未切 chunk)
    後面的 Ingestion pipeline 會接收 Document，切成 chunk -> embedding -> 存入 Qdrant
    每個 Document 包含兩個部分: text、metadata
    Args:
        note_ids: 要轉換的筆記 ID 清單
            - 若 None 則轉換所有筆記 (第一次建立知識庫時用)
            - 若是傳入清單表示只載入指定筆記 (增量更新時用)
    """
    # 一開始都先取得所有筆記
    all_notes = fetch_all_notes()

    # 表示只載入指定筆記 (增量更新時用)
    if note_ids is not None:
        target_ids = set(note_ids)
        # 只對 note_ids 裡不重複的筆記進行處理
        note_to_load = [n for n in all_notes if n["id"] in target_ids]
    else:
        note_to_load = all_notes  # 載入全部筆記

    documents = []

    for note in note_to_load:
        # === 這邊都在做 metadata 的處理 ===
        note_id = note["id"]
        title = note.get("title", "(無標題) ")

        # tags 在 HackMD API 回傳的是 list，例如 ["python", "RAG"]
        # 存進 metadata 時轉成逗號分隔的字串，方便 Qdrant payload 儲存與顯示

        tags = note.get("tags", [])
        tags_str = ", ".join(tags) if tags else ""

        # 取得該篇筆記的最後更新時間
        lastChangedAt = _parse_hackmd_timestamp(note["lastChangedAt"])

        print(f"  → 載入：{title}（{note_id}）")

        # === 這邊開始取得該筆記的內文 ===
        content = fetch_note_content(note_id)

        if not content.strip():
            print(f"  跳過空筆記: {title}")
            continue

        # 接下來開始 Document 的物件轉換
        doc = Document(
            text=content,
            metadata={
                "note_id": note_id,
                "title": title,
                "tags": tags_str,
                "source": "hackmd",
                "lastChangedAt": lastChangedAt,  # HackMD 上這篇筆記的最後更新時間
                "ingested_at": datetime.now().isoformat(),  # 這個 Document 被載入 Qdrant 的時間
            },
            doc_id=note_id,   # 用 note_id 作為 doc_id，確保同一篇筆記不會有重複 embedding
        )

        documents.append(doc)
    
    print(f"\n[HackMD] 成功載入 {len(documents)} 篇筆記為 Document 物件")
    return documents

if __name__ == "__main__":
    print("=== 測試 HackMD Loader ===\n")
    
    # 測試 1：取得所有筆記清單
    print("【測試 1】取得所有筆記 metadata 清單")
    notes = fetch_all_notes()
    for note in notes[:5]:   # 只印前 5 篇避免太長
        lastChangedAt= _parse_hackmd_timestamp(note["lastChangedAt"])
        print(f"  - {note['title']:<30} id={note['id']}  lastChangedAt={lastChangedAt}")
    
    print()
    
    # 測試 2：載入所有筆記為 Document 物件
    print("【測試 2】載入所有筆記為 Document 物件")
    docs = load_notes_as_documents()
    print(f"\n總共載入 {len(docs)} 篇，第一篇預覽：")
    if docs:
        print(f"  title   : {docs[0].metadata['title']}")
        print(f"  note_id : {docs[0].metadata['note_id']}")
        print(f"  tags    : {docs[0].metadata['tags']}")
        print(f"  內文前 100 字：{docs[0].text[:100]}...")