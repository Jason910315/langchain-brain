from rag.indexer import collection_exists, build_index, load_index
from rag.pipeline import build_query_engine, query_with_sources
from ingestion.doc_loader import load_documents


def main():
    print("=== LangChain Brain RAG 系統 ===\n")

    if collection_exists():
        print("找到既有 index，直接載入...")
        index = load_index()
    else:
        print("尚無 index，開始從 Google Drive 載入文件並建立 index...")
        documents = load_documents()
        if not documents:
            print("沒有找到任何文件，請確認 Google Drive 資料夾設定。")
            return
        index = build_index(documents)

    engine = build_query_engine(index)
    print("\n已就緒，輸入問題開始查詢（輸入 exit 離開）\n")

    while True:
        try:
            question = input("問題> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再見！")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print("再見！")
            break

        response = query_with_sources(engine, question)
        print(response)

if __name__ == "__main__":
    main()