import sys
from pathlib import Path, PurePosixPath
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from ingestion.doc_loader import get_drive_service, list_files_in_folder, filter_target_files, download_file, ROOT_FOLDER_ID
from rag.indexer import _get_qdrant_client, collection_exists, build_index, load_index, insert_documents
from llama_index.core.schema import Document, TextNode

from config import COLLECTION_NAME

"""
每次啟動時掃描 Drive、比對 Qdrant，檢查是否有新文件更新，對新文件執行 ingestion。整體流程：

  Drive file_id 集合  →  取差集  →  只下載並 ingest 新文件
  Qdrant doc_id 集合  ↗
"""

def get_existing_doc_ids() -> set[str]:
    """
    從 Qdrant 撈回所有已 Ingest 的文件的 doc_id (即 Google Drive file_id)
    流程: 撈出所有 chunk -> 判斷每個 chunk 在哪個文件 (doc_id) -> 這樣就可以知道哪些文件已經被處理
    """

    # 如果 COLLECTION_NAME 的空間不存在，會發生在初次啟動系統，還未建立任何 collection
    if not collection_exists():
        return set()

    client = _get_qdrant_client()
    doc_ids = set()
    offset = None

    while True:
        # 遞迴撈出所有 chunk (node) 的資料，points 就是 collection 內一個 node 的資料
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,         # Qdrant 撈資料要分批，每批的大小
            offset=offset,       # 下一批次要從哪裡開始撈
            with_payload=True,
            with_vectors=False,  # 不用向量資料
        )

        for point in points:
            # node 的原始文本
            node_content = point.payload.get("_node_content")
            if node_content:
                node = TextNode.from_json(node_content)
                # 將該 chunk 來自哪個文件 (file_id) 存進 doc_ids 集合
                if node.ref_doc_id:
                    doc_ids.add(node.ref_doc_id)

        # 沒有下一筆資料了，跳出迴圈
        if next_offset is None:
            break

        offset = next_offset

    return doc_ids

def detect_new_docs():
    """
    掃描 Google Drive 並與 Qdrant 比對，判斷是否有新文件
    """
    # 1. 掃描 Google Drive 上所有文件，掃下來可能有一個或多個新文件
    print("掃描 Google Drive...")
    service = get_drive_service()
    all_files = list_files_in_folder(service, ROOT_FOLDER_ID)
    # 我們過濾後要的目標檔案
    target_files = filter_target_files(all_files)
    print(f"Drive 內共 {len(target_files)} 份目標文件")

    # 2. 取得已存在於 Qdrant 的 doc_id 集合
    existing_ids = get_existing_doc_ids()
    print(f"Qdrant 上已有 {len(existing_ids)} 份文件")

    # 3. 找出新文件
    drive_ids = {f["id"] for f in target_files}
    new_ids = drive_ids - existing_ids

    if not new_ids:
        print("未發現新文件，不需 Ingestion")
        return 0
    
    new_files = [f for f in target_files if f["id"] in new_ids]
    return new_files

def sync_new_docs(new_files: list[dict], answer_llm: str):
    """
    將傳入的新文件清單下載每個檔案、切割、向量化後上傳 Qdrant
    流程: 第一次啟動 Qdrant 是空的 → build_index，之後每次執行 → load_index + insert_documents
    """
    print(f"發現 {len(new_files)} 份新文件，開始 Ingestion...")
    service = get_drive_service()

    # 遞迴下載所有新文件並轉為 Document 物件
    documents = []
    for i, file_info in enumerate(new_files, start=1):
        print(f"[{i}/{len(new_files)}] 下載: {file_info['path']}")
        try:
            content = download_file(service, file_info["id"])
            content = content.replace("\u200b", "").replace("\ufeff", "")  # 清隱形字元
            content = content.strip()                                      # 清頭尾空白

            if len(content) < 50:
                print("跳過 (內容太少)")
                continue
            source_folder = PurePosixPath(file_info['path']).parts[0]
            doc = Document(
                text=content,
                metadata={
                    "file_path": file_info['path'],
                    "file_name": file_info['name'],
                    "source_folder": source_folder,
                    "drive_file_id": file_info['id'],
                    "ingested_at": datetime.now().isoformat(),
                },
                doc_id=file_info['id'],
            )
            documents.append(doc)
        except Exception as e:
            print(f"錯誤: {e}，跳過此檔案")
            continue

    if not documents:
        return 0

    # 如果 COLLECTION_NAME 的空間不存在，代表是第一次建立，因此要 build_index，否則用 load_index 就好
    if collection_exists():
        index = load_index(answer_llm)
        insert_documents(index, documents)
    else:
        build_index(documents, answer_llm)

    print(f"同步完成，本次新增 {len(documents)} 份文件到 {COLLECTION_NAME} 空間中")
    return len(documents)


