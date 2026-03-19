import os
from typing import Iterable, List, Tuple

import chromadb
from fastembed import TextEmbedding
from chromadb.config import Settings

# PDF support
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


# ---------------- CONFIG ---------------- #
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "resume_chunks"
INPUT_DATA_DIR = os.path.join(os.path.dirname(__file__), "input-data")
ENCODE_BATCH_SIZE = 32
DB_ADD_BATCH_SIZE = 100
# ---------------------------------------- #


# ---------------- TEXT CHUNKING ---------------- #
def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """Splits text into smaller chunks for embedding."""
    text = text.strip()
    if not text:
        return []

    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) > max_chars:
            flush()
            current = para
        else:
            current += "\n\n" + para if current else para

    flush()
    return chunks
# ------------------------------------------------ #


# ---------------- FILE READING ---------------- #
def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(file_path: str) -> str:
    if PyPDF2 is None:
        print(f"⚠️ PyPDF2 not installed, skipping PDF: {file_path}")
        return ""

    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"⚠️ Error reading PDF {file_path}: {e}")

    return text


def read_file(file_path: str) -> str:
    if file_path.endswith(".txt") or file_path.endswith(".md"):
        return read_txt(file_path)
    elif file_path.endswith(".pdf"):
        return read_pdf(file_path)
    else:
        print(f"⚠️ Unsupported file skipped: {file_path}")
        return ""
# ------------------------------------------------ #


def _iter_input_files(input_dir: str) -> Iterable[str]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"❌ Input directory not found: {input_dir}")

    for root, _, files in os.walk(input_dir):
        for name in files:
            yield os.path.join(root, name)


# ---------------- LOAD DATA ---------------- #
def load_input_chunks(input_dir: str) -> Tuple[List[str], List[dict]]:
    all_chunks: List[str] = []
    all_metadatas: List[dict] = []

    input_dir = os.path.abspath(input_dir)
    files = list(_iter_input_files(input_dir))

    if not files:
        raise ValueError(f"❌ No files found in: {input_dir}")

    for file_path in files:
        text = read_file(file_path)

        if not text.strip():
            print(f"⚠️ Empty file skipped: {file_path}")
            continue

        chunks = chunk_text(text)

        if not chunks:
            print(f"⚠️ No chunks created from: {file_path}")
            continue

        rel_source = os.path.relpath(file_path, input_dir)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": rel_source,
                "chunk_index": i
            })

    if not all_chunks:
        raise ValueError("❌ No valid text chunks created. Add proper text files.")

    print(f"✅ Loaded {len(files)} files → {len(all_chunks)} chunks")
    return all_chunks, all_metadatas
# ------------------------------------------------ #


# ---------------- EMBEDDINGS ---------------- #
def compute_embeddings(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks."""
    model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    embeddings: List[List[float]] = []

    print("🔄 Generating embeddings...")
    for i in range(0, len(chunks), ENCODE_BATCH_SIZE):
        batch = chunks[i:i + ENCODE_BATCH_SIZE]
        for emb in model.embed(batch):
            embeddings.append(emb.tolist())

    print("✅ Embeddings created")
    return embeddings
# ------------------------------------------------ #


# ---------------- SAVE TO DB ---------------- #
def save_to_vector_store(chunks, embeddings, metadatas):
    # ✅ CRITICAL CHECK (Issue 15 fix)
    if len(chunks) != len(embeddings) or len(chunks) != len(metadatas):
        raise ValueError("Mismatch between chunks, embeddings, and metadata lengths")

    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)

    print("💾 Saving to database...")
    for i in range(0, len(chunks), DB_ADD_BATCH_SIZE):
        end = min(i + DB_ADD_BATCH_SIZE, len(chunks))

        # ✅ FIXED ID GENERATION (Issue 17)
        ids = [
            f"{metadatas[j].get('source','unknown')}::chunk-{metadatas[j].get('chunk_index', j)}"
            for j in range(i, end)
        ]

        collection.add(
            ids=ids,
            documents=chunks[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )

        print(f"  Stored {end}/{len(chunks)}")

    print("✅ Data stored in ChromaDB")
# ------------------------------------------------ #


# ---------------- MAIN ---------------- #
def main():
    chunks, metadatas = load_input_chunks(INPUT_DATA_DIR)
    embeddings = compute_embeddings(chunks)
    save_to_vector_store(chunks, embeddings, metadatas)


if __name__ == "__main__":
    main()
# ------------------------------------------------ #