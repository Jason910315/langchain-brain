import sys
import os
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

"""
上傳訓練資料到 OpenAI 的 Fine-tuning API 進行微調，分為兩步
1. 先上傳檔案拿到 file_id
2. 用這個 file_id 建立訓練 job
"""

def upload_training_file(client: OpenAI, file_path: str) -> str:
    """
    上傳本地 jsonl 檔案到 OpenAI，回傳 file_id
    """
    print(f"上傳訓練資料: {file_path}...")
    with open(file_path, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune",
        )

        file_id = response.id
        print(f"上傳成功，file_id: {file_id}")
        return file_id

def create_fine_tuning_job(client: OpenAI, file_id: str) -> str:
    """
    建立 Fine-tuning Job，回傳 job_id
    """
    print("建立 Fine-tuning Job...")

    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",
        method={
            "type": "supervised",
            "supervised": {
                # 微調過程的超參數，n_epochs 代表訓練幾輪
                "hyperparameters": {
                    "n_epochs": "3",
                }
            }
        },
    )

    job_id = job.id
    print(f"Fine-tuning job 建立成功，job_id: {job_id}")
    return job_id

def check_job_status(client: OpenAI, job_id: str) -> str:
    """
    讓你在訓練過程中，可以透過終端機執行指令查詢 Job 狀態
    """
    # 先找到剛剛我們建立的微調 job
    job = client.fine_tuning.jobs.retrieve(job_id)
    print(f"\n=== Fine-tuning Job 狀態 ===")
    print(f"job_id: {job_id}")
    print(f"status: {job.status}")
    print(f"model: {job.model}")

    if job.status == "succeeded":
        print("\n✅ 訓練完成!")
        print(f"Fine-tuning model name: {job.fine_tuned_model}")  # 訓練完成會自動建立這個微調模型的名字

    elif job.status == "failed":
        print("\n❌ 訓練失敗!")
        print(f"error: {job.error}")

    elif job.status in ["running", "queued", "validating_files"]:
        print(f"\n⏳ 訓練進行中，請稍後再查詢")

    # 印出最近的訓練資訊 (loss 等)
    events = client.fine_tuning.jobs.list_events(job_id, limit=5)
    if events.data:
        print("\n最近訓練資訊:")
        # 要 reverse 成反序，這樣才會是由舊到新
        for event in reversed(events.data):
            print(f"    {event.created_at}: {event.message}")

# 將微調資訊存到本地
def save_job_info(job_id: str, file_id: str):
    info_path = Path("data/ft_job_info.json")
    with open(info_path, "w") as f:
        json.dump({
            "jod_id": job_id,
            "file_id": file_id,
        }, f, ensure_ascii=False, indent=2)

        print(f"\njob 資訊已儲存至 {info_path}")


def main():
    # 原本執行 ft_data_generator.py 時，會開始微調
    # 這裡在額外定義參數，讓我們可以另外透過傳入參數執行腳本，來查詢 Job 狀態
    import argparse
    parser = argparse.ArgumentParser()
    # 添加 --status 參數，用來查詢指定 job_id 的訓練狀態
    parser.add_argument("--status", type=str, help="查詢指定 job_id 的訓練狀態")
    args = parser.parse_args()  # 解析每次執行腳本的參數

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 代表有帶 --status 參數 → 查詢進度
    # 可以查到訓練完後的模型名稱，後續要用它來做為回答模型
    if args.status:
        check_job_status(client, args.status)
        return

    train_path = Path("data/ft_train.jsonl")

    # 1. 上傳資料集檔案
    file_id = upload_training_file(client, train_path)

    # 2. 建立 Fine-tuning Job
    job_id = create_fine_tuning_job(client, file_id)

    # 3. 儲存 Job 資訊
    save_job_info(job_id, file_id)

    print(f"\n訓練開始，通常需要 10-30 分鐘")
    print(f"完成後用以下指令查詢結果：")
    print(f"   python ft_uploader.py --status {job_id}")

if __name__ == "__main__":
    # 如果有帶 --status 參數，則是在訓練過程中查詢狀態
    # 如果未帶 --status 參數，則是完整執行訓練流程
    main()
    


