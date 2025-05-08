import argparse
from dotenv import load_dotenv
from pre_process_data import PreProcessing
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_core.vectorstores import VectorStoreRetriever

def main():
    parser = argparse.ArgumentParser(description="事前処理済テキストをルールベースで前処理してチャンク化")
    parser.add_argument("input_file", help="問題テキストファイルのパス")
    parser.add_argument("input_file2", help="質問テキストファイルのパス")
    parser.add_argument("--faiss_dir", help="FAISSインデックスの出力パス(default: ./meta)", default="./meta")
    args = parser.parse_args()

    with open(args.input_file, encoding="utf-8") as f:
        raw_text = f.read()
    with open(args.input_file2, encoding="utf-8") as f:
        query = f.read()

    pp = PreProcessing()
    chunks = pp.preprocess_ocr_text(raw_text)
    vectorstore = pp.build_faiss(chunks, index_path=args.faiss_dir)

    load_dotenv(dotenv_path=".env.local", override=True)
    retriever = VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"k": 3})
    retrievalQA = RetrievalQA.from_llm(llm=OpenAI(model_name="gpt-3.5-turbo-instruct"), retriever=retriever)
    ans = retrievalQA.invoke({"query": query})
    print(ans['result'])

if __name__ == "__main__":
    main()
