# RAG Toy Project: ドキュメントベースの Q&A システム

韓国の情報サイト「Namuwiki」の「RAG」に関するページを参照データとして活用し、RAGの動作原理を学習・実装しました。

---

<br>

### 📌 プロジェクトの概要と目的
* **RAG動作原理の習得**: Load、Chunking、Embedding、Retrievalに至るRAGパイプライン全過程の概念を理解し、実際にコードを書いて実装しました。
* **プロンプト上書きとLLMの制御**: Groq APIベースのLLMに検索されたコンテキスト（`context`）を注入し、提供されたドキュメントの内容のみに基づいて回答するようハルシネーション（Hallucination）を制御しました。

<br>

## 🔍 Live Demo

> 🔗 **Live Demo:** [RAG Webアプリはこちら](https://ask-llm-anything.streamlit.app/)

<br>

## 🛠️ 技術スタック
* **UI**: Streamlit
* **Framework**: LangChain (LCEL)
* **LLM**: Groq API (`groq/compound-mini`)
* **Vector DB**: FAISS
* **Embedding Model**: HuggingFace (`jhgan/ko-sroberta-multitask`)

<br>



## 🚀 学習記録および性能改善

2026-08-30
* **LangChain構文の熟達**: LangChainモジュールとパイプ（`|`）演算子ベースのデータフロー処理、およびLangChainパッケージの構文にさらに習熟する。
* **検索性能の向上**: 単語キーワード検索（BM25）と Semantic検索（Vector DB）を組み合わせたハイブリッド検索（Hybrid Search）の学習および適用。
* **回答品質の改善**: 検索されたドキュメントの順位を再評価するリランキング（Reranking）の学習および適用。

2026-08-31
* **文法の構造的理解**: 単にコードを写経するだけの学習から脱却し、Pythonの基本的なオブジェクト指向概念であるパッケージ・モジュール・クラス・メソッドの構造をベースに、LangChainの動作原理を解剖するように分析・学習しました。<p>RAGの中核である前処理において、PDFのロード・分割から韓国語埋め込み、FAISS DB保存に至るデータ変換フローを把握しました。クラスの引数設定やメソッド呼び出しの構造を理解し、前処理体系を整理できました。<p>また、Pythonの命名規則から外部ライブラリのコードの正体を直感的に読み解く視点も身につけました。

* **オブジェクト指向の必要性**: 多くのPythonライブラリがオブジェクト指向で設計されていることを実感し、オブジェクト指向の概念を深く学び、基礎を確固たるものにしようと考えました。