from datetime import datetime
import io, os
from pathlib import PurePosixPath
from pathlib import Path, PurePosixPath
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from llama_index.core.schema import Document
from dotenv import load_dotenv

load_dotenv()

# Google Drive 根資料夾 ID
ROOT_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

BASE_DIR = Path(__file__).parent.parent
SERVICE_ACCOUNT_FILE = BASE_DIR / "ai-playground-chain-docs-reader.json"

# drive 上作為載入目標的起始資料夾
TARGET_FOLDERS = {"oss", "langsmith"}
# 支援的附檔名，要載入這類的資料檔
SUPPORTED_EXTENSION = {".mdx", ".md"}

def get_drive_service():
    """
    建立並回傳 Google Drive API 服務物件
    """
    # 指定服務帳號只有唯讀權限
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=scopes,
    )
    service = build("drive", "v3", credentials=credentials)
    return service

def list_files_in_folder(service, folder_id: str, current_path: str = "") -> list[dict]:
    """
    遞迴列出資料夾下的所有檔案
    Aegs:
        current_path: 一開始為空，進入 TARGET_FOLDERS 後才開始累加路徑
    """
    files = []
    page_token = None  # 當前頁面 (從第一頁開始)
    while True:
        # 找出父資料夾 id 是 folder_id 且不是在垃圾桶的 = (找出當前資料夾下的所有東西)
        query = f"'{folder_id}' in parents and trashed = false"
        response = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",  # 要回傳的欄位
            pageToken=page_token,  # 這次從哪一頁開始
            pageSize=1000,         # 每頁最多 1000 筆
        ).execute()

        items = response.get("files", [])

        # 要不斷取出
        for item in items:
            # name 為取到的資料夾 (檔案) 名稱，這裡的 current_path 是當前資料夾的路徑
            item_path = f"{current_path}/{item['name']}" if current_path else item["name"]

            # 當下是資料夾，要遞迴
            if item["mimeType"] == "application/vnd.google-apps.folder":
                sub_files = list_files_in_folder(service, item["id"], item_path)
                files.extend(sub_files)  # 遞迴出來的結果繼續加到 files 中
            else:
                files.append({
                    "id": item["id"],
                    "name": item["name"],
                    "path": item_path
                })
        
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    
    return files

# 只保留目標資料夾下且副檔名符合設定的檔案
def filter_target_files(all_files: list[dict]) -> list[dict]:
    result = []
    for f in all_files:
        path = PurePosixPath(f["path"])
        parts = path.parts  # 把路徑拆成各個部分

        if not parts or parts[0] not in TARGET_FOLDERS:  # parts[0] 就是起始進來的資料夾
            continue

        suffix = path.suffix.lower()  # 副檔名
        if suffix not in SUPPORTED_EXTENSION:
            continue
    
        result.append(f)
    
    return result

# --- 下載 Dive 內檔案 ---

def download_file(service, file_id: str) -> str:
    """
    從 Google Drive 下載單一檔案，並回傳文字內容
    """
    # 建立下載請求物件 (要下載 file_id)
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()  # 檔案下載時是二進位

    # Google API 的串流下載方式，負責把檔案分塊下載，每次下載一塊就存進 buffer
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk() # 呼叫下載一塊資料

    # 下載完後把讀取位置移到開頭
    buffer.seek(0)
    return buffer.read().decode("utf-8", errors="replace")

def load_documents() -> list[Document]:
    """
    從 Google Drive 載入並下載所有目標文件，再將其轉換為 LlamaIndex Document 物件的清單
    """
    
    print("連接 Google Drive API...")
    service = get_drive_service()

    print(f"遍歷資料夾: {ROOT_FOLDER_ID}")
    # 先抓取所有資料夾下的檔案
    all_files = list_files_in_folder(service, ROOT_FOLDER_ID)
    print(f"共找到 {len(all_files)} 個檔案")

    target_files = filter_target_files(all_files)
    print(f"過濾後剩 {len(target_files)} 個檔案")

    # 單一 file_info 的結構為 {id: , name: , path: }

    documents = []
    for i, file_info in enumerate(target_files, start=1):
        print(f"[{i}/{len(target_files)}] 下載: {file_info['path']}")  # 展示下載進度

        try:
            content = download_file(service, file_info["id"]) # 下載並返回完整檔案文字
            content = content.replace("\u200b", "").replace("\ufeff", "")  # 清隱形字元
            content = content.strip()                                      # 清頭尾空白

            if len(content) < 50:
                print("跳過 (內容太少)")
                continue
            
            # 他來自哪個起始資料夾
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

    print(f"\n載入完成，共 {len(documents)} 份文件")

    return documents

if __name__ == "__main__":

    docs = load_documents()

    if docs:
        print("\n--- 第一份文件預覽 ---")
        first = docs[0]
        print(f"路徑：{first.metadata['file_path']}")
        print(f"來源：{first.metadata['source_folder']}")
        print(f"字元數：{len(first.text)}")
        print(f"內容前 300 字：\n{first.text[:300]}")
    else:
        print("沒有載入任何文件，請確認 Drive 資料夾結構和權限設定")