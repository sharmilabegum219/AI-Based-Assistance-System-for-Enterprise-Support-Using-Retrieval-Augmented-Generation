"""
MSME RAG Pipeline — Flask Backend (Multilingual Version)
Supports: Telugu, Hindi, English + Romanized Telugu

KEY FIX — Per-chunk language detection:
  A single PDF can have both English and Telugu content.
  Every chunk is tagged with ITS OWN language, not the PDF's language.
  So a Telugu query searches Telugu chunks even inside an English PDF.

ARCHITECTURE:
  Telugu  query → search chunks where chunk_lang=telugu  → answer in Telugu
  Hindi   query → search chunks where chunk_lang=hindi   → answer in Hindi
  English query → search chunks where chunk_lang=english → answer in English
  Romanized Tel → search chunks where chunk_lang=telugu  → answer in Telugu script

Run: python app.py
"""

import os
import re
import uuid
import glob
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFium2Loader, PyPDFLoader
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import PromptTemplate

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PayloadSchemaType

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "msme_collection"
EMBED_MODEL     = "text-embedding-3-large"
DIMENSIONS      = 3072
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 120
TOP_K           = 7

DATA_ROOT = Path(r"C:\Users\hp\Downloads\govt_files\folder 6")

# Minimum Telugu/Hindi chars in a chunk to call it that language
SCRIPT_CHAR_THRESHOLD = 0.08   # 8% of chunk chars must be script chars


# ══════════════════════════════════════════════════════════════════════
#  PER-CHUNK LANGUAGE DETECTION  ← KEY FIX
# ══════════════════════════════════════════════════════════════════════

def detect_chunk_language(text: str) -> str:
    """
    Detect the language of a single chunk of text.
    Works correctly even when a PDF has mixed Telugu + English content.

    Returns: 'telugu' | 'hindi' | 'english'
    """
    total = max(len(text), 1)

    telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
    hindi_chars  = len(re.findall(r'[\u0900-\u097F]', text))

    if telugu_chars / total > SCRIPT_CHAR_THRESHOLD:
        return "telugu"
    if hindi_chars / total > SCRIPT_CHAR_THRESHOLD:
        return "hindi"
    return "english"


# ══════════════════════════════════════════════════════════════════════
#  USER QUERY LANGUAGE DETECTION  ← UPDATED: uses OpenAI for Latin text
# ══════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Detect language of user query using OpenAI for Latin-script text.
    Returns: 'telugu' | 'hindi' | 'english' | 'romanized_telugu'
    """
    text_stripped = text.strip()
    total_chars   = max(len(text_stripped), 1)

    # Fast script-based checks first (no API call needed)
    telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text_stripped))
    hindi_chars  = len(re.findall(r'[\u0900-\u097F]', text_stripped))

    if telugu_chars / total_chars > 0.2:
        return "telugu"
    if hindi_chars / total_chars > 0.2:
        return "hindi"

    # ── For Latin-script text, ask OpenAI ──────────────────────────
    try:
        response = llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a language detection assistant. "
                    "Classify the following text into EXACTLY ONE of these categories:\n"
                    "- 'romanized_telugu': Telugu language written using English/Latin letters "
                    "(e.g., 'mee peru emi', 'loan ela teesukovali', 'scheme gurinchi cheppandi')\n"
                    "- 'romanized_hindi': Hindi language written in English/Latin letters "
                    "(e.g., 'aapka naam kya hai', 'yojana ke baare mein batao')\n"
                    "- 'english': Actual English language\n\n"
                    "Reply with ONLY one word: romanized_telugu, romanized_hindi, or english. "
                    "No explanation, no punctuation."
                ),
            },
            {"role": "user", "content": text_stripped},
        ])
        detected = response.content.strip().lower()
        print(f"  🤖 OpenAI language detection: '{detected}'")

        if detected in ("romanized_telugu", "romanized_hindi", "english"):
            # Map romanized_hindi → hindi for downstream compatibility
            return "hindi" if detected == "romanized_hindi" else detected
        return "english"

    except Exception as e:
        print(f"  ⚠️  OpenAI language detection failed ({e}). Defaulting to 'english'.")
        return "english"


def get_chunk_language_for_query(lang: str) -> str:
    """Map user query language → chunk_lang tag to filter on."""
    return {
        "telugu":           "telugu",
        "romanized_telugu": "telugu",
        "hindi":            "hindi",
        "english":          "english",
    }.get(lang, "english")


# ══════════════════════════════════════════════════════════════════════
#  PDF HELPERS
# ══════════════════════════════════════════════════════════════════════

def list_pdfs_by_category(root: Path) -> dict:
    cats      = {}
    root_pdfs = sorted(glob.glob(str(root / "*.pdf")))
    if root_pdfs:
        cats["general"] = root_pdfs
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        pdfs = sorted(glob.glob(str(sub / "*.pdf")))
        if pdfs:
            cats[sub.name] = pdfs
    return cats


def load_pdf_safe(path: str) -> List[Document]:
    for Loader in (PyMuPDFLoader, PyPDFium2Loader, PyPDFLoader):
        try:
            docs = Loader(path).load()
            if docs:
                return docs
        except Exception:
            continue
    print(f"  ❌ All loaders failed: {path}")
    return []


def chunk_pdf(pdf_path: str, category: str) -> List[Document]:
    """
    Load, split, and tag every chunk with its own detected language.

    This is the KEY FIX:
      - Old code: tagged the whole PDF as one language
      - New code: tags EACH CHUNK individually

    So an English PDF with Telugu paragraphs will produce:
      chunk_0: chunk_lang=english
      chunk_1: chunk_lang=telugu   ← Telugu query will find this
      chunk_2: chunk_lang=english
      chunk_3: chunk_lang=telugu   ← and this
    """
    docs = load_pdf_safe(pdf_path)
    if not docs:
        return []

    # Use mixed separators so both Telugu and English split well
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "।", ".", " ", ""],
    )

    raw_chunks = splitter.split_documents(docs)
    name       = Path(pdf_path).stem

    lang_counts = {"telugu": 0, "hindi": 0, "english": 0}

    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        # Detect language for THIS chunk individually
        chunk_lang = detect_chunk_language(chunk.page_content)
        lang_counts[chunk_lang] += 1

        chunk.metadata.update({
            "category":    category,
            "pdf_name":    name,
            "chunk_index": i,
            "source_path": pdf_path,
            "chunk_lang":  chunk_lang,    # ← per-chunk language tag
        })
        final_chunks.append(chunk)

    print(
        f"  📄 {Path(pdf_path).name} → {len(final_chunks)} chunks  "
        f"(tel={lang_counts['telugu']} | hin={lang_counts['hindi']} | eng={lang_counts['english']})"
    )
    return final_chunks


def make_ids(chunks: List[Document]) -> List[str]:
    return [
        str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{d.metadata['pdf_name']}|{d.metadata['chunk_index']}"
        ))
        for d in chunks
    ]


# ══════════════════════════════════════════════════════════════════════
#  QDRANT SETUP
# ══════════════════════════════════════════════════════════════════════

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# Create collection if missing
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=DIMENSIONS,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"✅ Created collection: {COLLECTION_NAME}")

# Ensure payload index on chunk_lang (runs every startup — safe to repeat)
try:
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chunk_lang",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("✅ Payload index on 'chunk_lang' ensured.")
except Exception as e:
    print(f"ℹ️  Payload index already present (OK): {e}")

embeddings  = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY,
)


# ══════════════════════════════════════════════════════════════════════
#  RETRIEVAL — filter on chunk_lang
# ══════════════════════════════════════════════════════════════════════

def retrieve_by_chunk_language(query: str, chunk_lang: str, k: int = TOP_K) -> List[Document]:
    """
    Search only chunks tagged with chunk_lang.
    Falls back to unfiltered search if no results found.
    """
    lang_filter = Filter(
        must=[
            FieldCondition(
                key="chunk_lang",
                match=MatchValue(value=chunk_lang),
            )
        ]
    )

    # Filtered search
    try:
        results = vectorstore.similarity_search(query, k=k, filter=lang_filter)
        if results:
            print(f"  ✅ {len(results)} '{chunk_lang}' chunks retrieved (filtered)")
            return results
        print(f"  ⚠️  0 filtered results for chunk_lang='{chunk_lang}' — trying unfiltered")
    except Exception as e:
        print(f"  ⚠️  Filtered search error: {e} — trying unfiltered")

    # Unfiltered fallback
    try:
        results = vectorstore.similarity_search(query, k=k)
        print(f"  ↩️  {len(results)} chunks retrieved (unfiltered fallback)")
        return results
    except Exception as e:
        print(f"  ❌ Unfiltered search also failed: {e}")
        return []


def translate_romanized_to_telugu_script(query: str) -> str:
    """Convert Romanized Telugu → Telugu script for better vector match."""
    try:
        response = llm.invoke([
            {
                "role": "system",
                "content": (
                    "Convert the following Romanized Telugu (Telugu written in English letters) "
                    "into proper Telugu script. Return ONLY the Telugu script, nothing else."
                ),
            },
            {"role": "user", "content": query},
        ])
        converted = response.content.strip()
        print(f"  🔄 Romanized → Telugu script: {converted}")
        return converted
    except Exception as e:
        print(f"  ⚠️  Conversion failed ({e}). Using original query.")
        return query


# ══════════════════════════════════════════════════════════════════════
#  PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_multilingual_prompt(lang: str) -> PromptTemplate:
    if lang == "telugu":
        instruction = (
            "మీరు భారతదేశంలో MSME పథకాలు మరియు విధానాలపై నిపుణుడైన సహాయకుడు.\n"
            "దిగువ context ఆధారంగా మాత్రమే సమాధానం ఇవ్వండి.\n"
            "సమాధానం తెలుగులో స్పష్టంగా, నిర్మాణాత్మకంగా ఇవ్వండి.\n"
            "Context లో సమాచారం లేకపోతే: "
            "'నా knowledge base లో ఈ విషయమై తగినంత సమాచారం లేదు.' అని చెప్పండి.\n\n"
            "Context:\n{context}\n\nప్రశ్న: {question}\n\nసమాధానం (తెలుగులో):"
        )
    elif lang == "romanized_telugu":
        instruction = (
            "You are an expert assistant on MSME schemes and policies in India.\n"
            "The user asked in Romanized Telugu. Answer in Telugu script.\n"
            "Use ONLY the context below.\n"
            "If not in context: 'మా knowledge base లో ఈ విషయంపై సమాచారం లేదు.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer (in Telugu script):"
        )
    elif lang == "hindi":
        instruction = (
            "आप MSME योजनाओं और नीतियों के विशेषज्ञ सहायक हैं।\n"
            "नीचे दिए गए context के आधार पर ही उत्तर दें। उत्तर हिंदी में दें।\n"
            "यदि context में नहीं है: 'मेरे knowledge base में इस विषय पर जानकारी नहीं है।'\n\n"
            "Context:\n{context}\n\nप्रश्न: {question}\n\nउत्तर (हिंदी में):"
        )
    else:
        instruction = (
            "You are an expert assistant on MSME schemes and policies in India.\n"
            "Use ONLY the context below. Be clear and use bullet points where helpful.\n"
            "If not in context: 'I don't have enough information about this.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
    return PromptTemplate(template=instruction, input_variables=["context", "question"])


# ══════════════════════════════════════════════════════════════════════
#  SMART QA
# ══════════════════════════════════════════════════════════════════════

def smart_qa(query: str) -> dict:
    """
    Full pipeline:
      1. Detect user query language (OpenAI for Latin-script text)
      2. Prepare search query (Romanized→Telugu script if needed)
      3. Retrieve chunks filtered by chunk_lang
      4. Answer in the original query language
    """

    # Step 1 — Detect query language
    lang = detect_language(query)
    print(f"\n🌐 Detected language: {lang}")

    # Step 2 — Prepare search query
    if lang == "romanized_telugu":
        search_query = translate_romanized_to_telugu_script(query)
    else:
        search_query = query   # Telugu/Hindi/English embeddings work natively

    # Step 3 — Determine chunk_lang to filter on
    chunk_lang = get_chunk_language_for_query(lang)
    print(f"  🔍 Searching chunks with chunk_lang='{chunk_lang}'...")

    # Step 4 — Retrieve
    retrieved_docs = retrieve_by_chunk_language(search_query, chunk_lang, k=TOP_K)

    if not retrieved_docs:
        fallback = {
            "telugu":           "సంబంధిత సమాచారం కనుగొనబడలేదు. దయచేసి మీ ప్రశ్నను మరింత స్పష్టంగా అడగండి.",
            "romanized_telugu": "Sambandhitha samacharam kanugonabadaleda. Dayachesi mee prashnanu marinto spashtanga adagandi.",
            "hindi":            "संबंधित जानकारी नहीं मिली। कृपया प्रश्न स्पष्ट करें।",
            "english":          "No relevant information found. Please rephrase your question.",
        }
        return {
            "answer":            fallback.get(lang, fallback["english"]),
            "sources":           [],
            "detected_language": lang,
        }

    # Step 5 — Build context + answer
    context         = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt_template = build_multilingual_prompt(lang)
    prompt_text     = prompt_template.format(context=context, question=query)
    response        = llm.invoke([{"role": "user", "content": prompt_text}])
    answer          = response.content.strip()

    # Step 6 — Deduplicate sources
    seen    = set()
    sources = []
    for doc in retrieved_docs:
        key = doc.metadata.get("pdf_name", "")
        if key not in seen:
            seen.add(key)
            sources.append({
                "pdf_name":    doc.metadata.get("pdf_name", "unknown"),
                "category":    doc.metadata.get("category", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "chunk_lang":  doc.metadata.get("chunk_lang", "english"),
            })

    return {
        "answer":            answer,
        "sources":           sources,
        "detected_language": lang,
    }


# ══════════════════════════════════════════════════════════════════════
#  AUTO INGEST
# ══════════════════════════════════════════════════════════════════════

def auto_ingest():
    print("\n🔄 AUTO INGEST CHECK...")
    cats  = list_pdfs_by_category(DATA_ROOT)
    total = 0
    for category, pdfs in cats.items():
        for pdf in pdfs:
            chunks = chunk_pdf(pdf, category)
            if not chunks:
                continue
            ids      = make_ids(chunks)
            existing = qdrant_client.retrieve(collection_name=COLLECTION_NAME, ids=ids)
            if existing:
                print(f"  ⏩ Already indexed: {Path(pdf).name}")
                continue
            vectorstore.add_documents(documents=chunks, ids=ids)
            total += len(chunks)
            print(f"  ✔ Ingested: {Path(pdf).name} ({len(chunks)} chunks)")
    print(f"\n✅ Auto-ingest done. Added {total} new chunks.\n")


auto_ingest()


# ══════════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder=".")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "msme_frontend.html")


@app.route("/api/info", methods=["GET"])
def api_info():
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        cats = list_pdfs_by_category(DATA_ROOT)
        return jsonify({
            "collection":          COLLECTION_NAME,
            "points":              info.points_count,
            "embed_model":         EMBED_MODEL,
            "categories":          {n: {"pdf_count": len(p)} for n, p in cats.items()},
            "languages_supported": ["English", "Telugu", "Hindi", "Romanized Telugu"],
            "retrieval_strategy":  "per-chunk language detection + filtered retrieval + OpenAI lang detect",
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body  = request.json or {}
    query = body.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"})
    try:
        return jsonify(smart_qa(query))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/api/detect_language", methods=["POST"])
def api_detect_language():
    body  = request.json or {}
    query = body.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"})
    lang       = detect_language(query)
    chunk_lang = get_chunk_language_for_query(lang)
    return jsonify({
        "query":              query,
        "detected_language":  lang,
        "will_search_chunks": chunk_lang,
    })


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    try:
        cats        = list_pdfs_by_category(DATA_ROOT)
        per_cat     = {}
        grand_total = 0
        failed      = []
        for category, pdfs in cats.items():
            cat_chunks = 0
            for pdf in pdfs:
                chunks = chunk_pdf(pdf, category)
                if not chunks:
                    failed.append({"category": category, "file": Path(pdf).name, "error": "load failed"})
                    continue
                ids = make_ids(chunks)
                vectorstore.add_documents(documents=chunks, ids=ids)
                cat_chunks += len(chunks)
            per_cat[category] = cat_chunks
            grand_total       += cat_chunks
        return jsonify({"grand_total": grand_total, "per_category": per_cat, "failed": failed})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
