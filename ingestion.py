
from pathlib import Path
import hashlib
import re
import shutil

import fitz
import pandas as pd
from docx import Document

from rag_core import load_resources, COURSES


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

ALLOWED_TYPES = {".pdf", ".docx", ".csv", ".txt"}


def clean_text(text):
    text = text.replace("\x00", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(path):
    extension = path.suffix.lower()

    if extension == ".pdf":
        with fitz.open(str(path)) as pdf:
            for page_no, page in enumerate(pdf, 1):
                yield page_no, page.get_text()

    elif extension == ".docx":
        document = Document(path)

        paragraphs = [
            p.text for p in document.paragraphs
        ]

        for table in document.tables:
            for row in table.rows:
                paragraphs.append(
                    " | ".join(cell.text for cell in row.cells)
                )

        yield -1, "\n".join(paragraphs)

    elif extension == ".csv":
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False
        )

        for row_no, (_, row) in enumerate(
            df.iterrows(), 1
        ):
            text = " | ".join(
                f"{column}: {value}"
                for column, value in row.items()
                if str(value).strip()
            )
            yield -1, f"Row {row_no}: {text}"

    elif extension == ".txt":
        yield -1, path.read_text(
            encoding="utf-8-sig"
        )


def split_text(text, chunk_size=180, overlap=30):
    words = clean_text(text).split()
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk:
            yield chunk

        if start + chunk_size >= len(words):
            break


def ingest_document(file_path, course):
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() not in ALLOWED_TYPES:
        raise ValueError("Unsupported file type.")

    if course not in COURSES.values():
        raise ValueError("Unknown course.")

    # Stable file identity prevents duplicate chunk IDs
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    safe_name = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        path.name
    )

    saved_name = f"{digest[:12]}_{safe_name}"

    records = []

    for page, text in extract_sections(path):
        for piece in split_text(text):
            records.append({
                "text": piece,
                "page": page
            })

    if not records:
        raise ValueError(
            "No extractable text found in this file."
        )

    model, collection = load_resources()

    vectors = model.encode(
        [r["text"] for r in records],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    destination = UPLOAD_DIR / course
    destination.mkdir(parents=True, exist_ok=True)

    saved_file = destination / saved_name

    if path.resolve() != saved_file.resolve():
        shutil.copy2(path, saved_file)

    before = collection.count()

    for start in range(0, len(records), 100):
        end = min(start + 100, len(records))
        batch = records[start:end]

        collection.upsert(
            ids=[
                f"upload_{course}_{digest[:16]}_{i}"
                for i in range(start, end)
            ],
            documents=[r["text"] for r in batch],
            embeddings=vectors[start:end].tolist(),
            metadatas=[
                {
                    "course": course,
                    "source": saved_name,
                    "type": path.suffix[1:].upper(),
                    "page": int(r["page"]),
                    "file_hash": digest,
                    "origin": "user_upload"
                }
                for r in batch
            ]
        )

    after = collection.count()

    return {
        "file": saved_name,
        "processed_chunks": len(records),
        "new_chunks": after - before,
        "total_chunks": after
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python ingestion.py "
            "<file_path> <course_id>"
        )
        sys.exit(1)

    result = ingest_document(
        sys.argv[1],
        sys.argv[2]
    )

    print("Ingestion completed!")
    print(result)
