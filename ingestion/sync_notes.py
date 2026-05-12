import sys, os
from pathlib import Path
from datetime import datetime, timezone

# 把專案根目錄加進 Python 路徑，讓這個檔案可以 import 根目錄的模組
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, setup_settings

from ingestion.hackmd_loader import fetch_all_notes, load_notes_as_documents
from rag.indexer import collection_exists, build_index, insert_documents, load_index


load_dotenv()

def _get_qdrant_client() -> QdrantClient:

    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY,)

def get_qdrant_note_index() -> dict[str, str]:
    """
    從 Qdrant 撈回所有 chunk，在建立 note_id 和 lastChangedAt 的對照表供後續比對使用
    Returns:
        - 該篇筆記在 Qdrant 內紀錄的 lastChangedAt 欄位的對照表，{"abc123": "2024-05-01T10:00:00+00:00", ...}
          (Qdrant 內的 lastChangedAt 會是那篇筆記那次更新後載入到 Qdrant，Qdrant 記錄他當下的最後更新時間，
          但 HackMD 中該筆記的 lastChangedAt 可能會一直更新，但還沒載到 Qdrant 中)
    """

    # 如果 COLLECTION_NAME 的空間不存在，會發生在初次啟動系統，還未建立任何 collection
    if not collection_exists():
        return {}

    client = client = _get_qdrant_client()
    note_index = {} # 最終回傳的對照表
    offset = None

    # 遞迴撈出所有 chunk (node) 的資料，points 就是 collection 內一個 node 的資料
    while True:
        # scroll() 每次回傳一批 points（chunk）和下一批的起點
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,         # Qdrant 撈資料要分批，每批的大小
            offset=offset,       # 下一批次要從哪裡開始撈
            with_payload=True,
            with_vectors=False,  # 不用向量資料
        )

        for point in points:
            payload = point.payload
            # 取出 Qdrant 內每個 chunk 的資訊 (這兩個資訊後面要來做筆記更新的比對)
            note_id = payload.get("note_id")
            lastChangedAt = payload.get("lastChangedAt")

            if note_id and lastChangedAt:
                # setdefault() 如果 note_id 不存在，則新增一筆 {note_id: lastChangedAt}
                # 因為同一篇筆記會有很多 chunk，這樣確保每個 note_id (那篇筆記) 只會紀錄一次，就是在第一次出現時紀錄
                note_index.setdefault(note_id, lastChangedAt)

        # 沒有下一筆資料了，跳出迴圈
        if next_offset is None:
            break

        offset = next_offset

    print(f"目前 Qdrant 知識庫已有 {len(note_index)} 篇筆記的資料")
    return note_index

def detect_changes(hackmd_notes: list[dict], qdrant_index: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """
    比對 HackMD 筆記的最新狀態與 Qdrant 內知識的現有狀態，找出需要處理 (hackmd 有更新但還未進入 Qdrant) 的筆記
    Returns: (new_notes, updated_notes)
        new_notes: HackMD 有、Qdrant 沒有的新筆記 → 需要直接 ingest
        updated_notes: 兩邊都有、但 HackMD 更新時間較新的筆記 → 需要先刪後建
        → 但其實兩者最後一步都是重建
    """

    new_notes = []
    updated_notes = []
    
    # 從 hackMD 的筆記清單內一筆一筆比對，note 的格式是一個 dict，裡面都是該筆記的 metadata
    for note in hackmd_notes:
        note_id = note["id"]
        title = note.get("title", "(無標題) ")

        # hackMD 的 lastChangedAt 是 Unix Timestamp 毫秒，需要轉換為 ISO 8601 格式才能跟 Qdrant 內資料比較
        lastChangedAt_at_ms = note.get("lastChangedAt", 0)
        lastChangedAt_at_iso = datetime.fromtimestamp(
            lastChangedAt_at_ms / 1000, tz=timezone.utc
        ).isoformat()

        # 情況 A：Qdrant 沒有這篇 → 新筆記，要 ingest
        if note_id not in qdrant_index:
            print(f" → [新增筆記]: {title}（{note_id}）")
            new_notes.append(note)

        # 情況 B：Qdrant 有，但 HackMD 的更新時間比較新 → 筆記被改過，要先刪後建
        elif lastChangedAt_at_iso > qdrant_index[note_id]:
            print(" → [更新筆記知識庫] {title} (HackMD: {lastChangedAt_at_iso} > Qdrant: {qdrant_index[note_id]})")
            updated_notes.append(note)

    return new_notes, updated_notes 

def delete_note_chunks(note_id: str) -> None:
    """
    從 Qdrant 內刪除指定 note_id 的所有 chunks (因為該筆記有更新或從未進入，就將該筆記所 chunks 全部刪除重建)
    """

    client = _get_qdrant_client()

    # 用 chunk 的 payload 的欄位值去做刪除
    # Filter + FieldCondition + MatchValue 組合起來等同於 SQL 的：
    # DELETE FROM collection WHERE note_id = '{note_id}'
    client.delete(
        collection_name=COLLECTION_NAME,
        point_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="note_id",      # chunk 裡的 payload 欄位
                        match=MatchValue(value=note_id),   # 要匹配的值
                    )
                ]
            )
        )
    )
    print(f"已刪除 {note_id} 的所有舊 chunks")

def sync_notes() -> dict:
    """
    主流程: 執行一次完整的 HackMD 和 Qdrant 的同步流程
    """
    print("開始 HackMD ↔ Qdrant 增量同步")
    print("=" * 50)

    # 1. 取得 HackMD 所有筆記清單
    print("\n從 HackMD 取得最新筆記清單...")
    hackmd_notes = fetch_all_notes()

    # 2. 取得 Qdrant 的 note、ingested_at 對照表
    print("\n從 Qdrant 取得現有筆記索引...")
    qdrant_index = get_qdrant_note_index()

    # 3. 比對差異
    print("\n比對差異，找出需要處理的筆記...")
    new_notes, updated_notes = detect_changes(hackmd_notes, qdrant_index)

    skipped = len(hackmd_notes) - len(new_notes) - len(updated_notes)
    print(f"\n比對結果：新增 {len(new_notes)} 篇 / 更新 {len(updated_notes)} 篇 / 跳過 {skipped} 篇")

    # 如果完全沒有需要處理的，直接結束
    if not new_notes and not updated_notes:
        print("\n✅ 知識庫已是最新，無需同步")
        return {"new": 0, "updated": 0, "skipped": skipped, "total": len(hackmd_notes)}

    # 4. 把要更新的筆記先刪掉舊 chunk（先刪後建的「先刪」）
    if updated_notes:
        print(f"\n刪除 {len(updated_notes)} 篇已更新筆記的舊 chunk...")
        for note in updated_notes:
            delete_note_chunks(note["id"])

    # 5. 載入需要 ingest 的筆記內文，送進 indexer
    notes_to_ingest = new_notes + updated_notes
    note_ids_to_ingest = [n["id"] for n in notes_to_ingest]

    print(f"\n載入 {len(notes_to_ingest)} 篇筆記並建立向量索引...")

    # 只抓需要 ingest 的筆記內文轉成 document 物件清單，不需要全部重抓
    documents = load_notes_as_documents(note_ids=note_ids_to_ingest)

    # collection 不存在時要用 build_index()，它會建立 collection 並存入資料
    if not collection_exists():
        print("  → 知識庫不存在，建立新的 collection...")
        build_index(documents, answer_llm="OpenAI/gpt-4o-mini")  # answer_llm 任意設，這裡只做了 embedding 而已
    else:
        print("  → 知識庫已存在，增量插入...")
        index = load_index(answer_llm="OpenAI/gpt-4o-mini")
        insert_documents(index, documents)

    print("\n✅ 同步完成！")
    print(f"   新增：{len(new_notes)} 篇")
    print(f"   更新：{len(updated_notes)} 篇")
    print(f"   跳過：{skipped} 篇（無變動）")

    return {
        "new": len(new_notes), "updated": len(updated_notes), "skipped": skipped, "total": len(hackmd_notes)
    }

# ── 單獨執行此檔案時，直接跑一次完整同步 ──
if __name__ == "__main__":
    # setup_settings 要在任何 LlamaIndex 操作之前呼叫，
    # 確保 embed model 已初始化，否則 ingest 時會找不到 embedding 模型
    setup_settings("OpenAI/gpt-4o-mini")
    result = sync_notes()
    print(f"\n最終結果：{result}")