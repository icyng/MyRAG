import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

"""
# 概要

- 表や図などをOCRあるいは手動でテキスト化したようなデータを機械が翻訳できるよう前処理する
- 入力された非構造化テキストに対し、半構造化データ("【{section}】\n{chunk}"←こんな感じ)にしてモデルに渡す役目を果たす

# 注意事項

- 前処理を走らせる際には、必ず以下の事前処理を行うことに留意：
    - テキストにおけるセクションの見出しは`【見出し】`形式に修正し、かつ先頭に`\n\n`を2つ挿入する（分割箇所を明確にするため）
    - 問題文中の「ア」のように空欄部を表す単語については、あらかじめ` [ア] `のように「角括弧 + 前後の空白」に修正しておく
    - 問題文中の傍線部で表される部分については、あらかじめ鉤括弧`{}`を用いた表示で囲むように修正しておく
    - 適宜、問題冊子のPDFデータをテキスト化した際に発生する不要な改行や全角空白を除去しておく（不要な場合もあり）
"""

class PreProcessing:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct", device: str = "mps"):
        """
        Args:
            model_name (str): モデル
            device (str): デバイス ("cpu", "cuda:0", "mps" など)
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )

    def detect_sections(self, text: str) -> dict[str, str]:
        """
        - 【ヘッダ】形式の見出し(かつ先頭に改行*2)でセクションごとに分割し、辞書で返す
        - 括弧内の文字列をキーに、以降テキストを値として保持
        """
        pattern = r"(?:\n{2}【([^】]+)】)"
        parts = re.split(pattern, text)
        sections: dict[str, str] = {}
        for i in range(1, len(parts), 2):
            key = parts[i]
            content = parts[i+1] if i+1 < len(parts) else ""
            sections[key] = content.strip()
        return sections

    def split_into_chunks(
        self,
        text: str,
        max_chars: int = 100,
        overlap: int = 50
    ) -> list[str]:
        """
        与えられたテキストをチャンク化および埋め込む
        """
        docs = [Document(page_content=text)]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "、", "．", "，", " "],
            length_function=len
        )
        chunks: list[Document] = text_splitter.split_documents(docs)
        return [doc.page_content for doc in chunks]

    def build_faiss(
        self,
        chunks: dict[str, list[str]],
        index_path: str = None
    ):
        """
        チャンクをFAISS形式のベクトルストアとして構築する

        Args:
            chunks (dict): セクションIDをキー、チャンク本文リストを値とする辞書
            index_path (str): インデックス保存先ディレクトリ（Noneなら保存しない）
        Returns:
            vectorstore: LangChain FAISSベクトルストア
        """
        docs = []
        for sec, texts in chunks.items():
            for text in texts:
                combined = f"【{sec}】\n{text}"
                docs.append(Document(page_content=combined, metadata={"section": sec}))
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        if index_path:
            vectorstore.save_local(index_path)
        return vectorstore

    def preprocess_ocr_text(self, raw_text: str) -> dict[str, list[str]]:
        """
        「セクション検出」と「各セクションごとのチャンク分割」をまとめたもの
        """
        secs = self.detect_sections(raw_text)
        result: dict[str, list[str]] = {}
        for sec, content in secs.items():
            result[sec] = self.split_into_chunks(content)
        return result