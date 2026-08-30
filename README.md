# RAG Toy Project: ドキュメントベースの Q&A システム

Streamlit、LangChain、Groq APIを活用して構築したRAG（検索拡張生成）のトイプロジェクトです。

---

<br>

### 📌 プロジェクトの概要と目的
* **RAG動作原理の習得**: Load、Chunking、Embedding、Retrievalに至るRAGパイプライン全過程の概念を理解し、実際にコードを書いて実装しました。
* **プロンプト上書きとLLMの制御**: Groq APIベースのLLMに検索されたコンテキスト（`context`）を注入し、提供されたドキュメントの内容のみに基づいて回答するようハルシネーション（Hallucination）を制御しました。

---

<br>

### 🔍 Live Demo

> 🔗 **Live Demo:** [RAG Webアプリはこちら](https://ask-llm-anything.streamlit.app/)

---

<br>

### 🛠️ 技術スタック
* **UI**: Streamlit
* **Framework**: LangChain (LCEL)
* **LLM**: Groq API (`groq/compound-mini`)
* **Vector DB**: FAISS
* **Embedding Model**: HuggingFace (`jhgan/ko-sroberta-multitask`)

---

<br>

### 🚀 性能改善および今後の学習方向性

2026-08-30
* **LangChain構文の熟達**: LangChainモジュールとパイプ（`|`）演算子ベースのデータフロー処理、およびLangChainパッケージの構文にさらに習熟する。
* **検索性能の向上**: 単語キーワード検索（BM25）と Semantic検索（Vector DB）を組み合わせたハイブリッド検索（Hybrid Search）の学習および適用。
* **回答品質の改善**: 検索されたドキュメントの順位を再評価するリランキング（Reranking）の学習および適用。