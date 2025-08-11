import logging
import json
from pathlib import Path
import re
from typing import List, Optional
import unicodedata
import fitz
import pdfplumber
import docx2txt
from bs4 import BeautifulSoup
import requests
import easyocr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from src.config import Config

logger = logging.getLogger(__name__)


class DocumentIngestor:
    def __init__(self, embedding_model: str = None):
        self.config = Config()
        self.embedding_model = embedding_model or self.config.EMBEDDING_MODEL
        self.embeddings = self._initialize_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
        )

    def _initialize_embeddings(self):
        if self.embedding_model == "huggingface":
            model_name = self.config.EMBEDDING_MODEL_NAME
            logger.info(f"Using HuggingFace embedding model: {model_name}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

    # ---------------- PDF ----------------
    def load_pdf_pymupdf(self, file_path: Path) -> str:
        try:
            doc = fitz.open(file_path)
            text = "".join([page.get_text() for page in doc])
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return ""

    def load_pdf_pdfplumber(self, file_path: Path) -> str:
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return ""

    # ---------------- Text / MD ----------------
    def load_text_file(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return ""

    # ---------------- DOCX ----------------
    def load_docx_file(self, file_path: Path) -> str:
        try:
            return docx2txt.process(str(file_path))
        except Exception as e:
            logger.error(f"Error loading DOCX file {file_path}: {e}")
            return ""

    # ---------------- HTML ----------------
    def load_html_file(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                return soup.get_text(separator="\n")
        except Exception as e:
            logger.error(f"Error loading HTML file {file_path}: {e}")
            return ""

    # ---------------- Image (EasyOCR) ----------------

    def load_image_file(self, file_path: Path) -> str:
        try:
            # Initialize EasyOCR reader (English only, you can add more languages if needed)
            reader = easyocr.Reader(["en"], gpu=False)
            results = reader.readtext(str(file_path), detail=0)
            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error extracting text from image {file_path}: {e}")
            return ""

    # ---------------- Web Scraper ----------------
    def load_web_pages(self) -> List[Document]:
        docs = []
        for url in self.config.WEB_URLS:
            try:
                html = requests.get(url, timeout=10).text
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n")
                docs.append(
                    Document(
                        page_content=text, metadata={"source": url, "file_type": "web"}
                    )
                )
                logger.info(f"Scraped: {url}")
            except Exception as e:
                logger.error(f"Error loading URL {url}: {e}")
        return docs

    # ---------------- Main Loader ----------------
    def load_document(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            content = self.load_pdf_pymupdf(file_path)
            if not content.strip():
                content = self.load_pdf_pdfplumber(file_path)
            return content

        elif suffix in [".txt", ".md", ".markdown"]:
            return self.load_text_file(file_path)

        elif suffix == ".docx":
            return self.load_docx_file(file_path)

        elif suffix in [".html", ".htm"]:
            return self.load_html_file(file_path)

        elif suffix in [".png", ".jpg", ".jpeg"]:
            return self.load_image_file(file_path)

        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return ""

    def clean_text(self, text: str) -> str:
        """
        Cleans noisy OCR and sensitive data from text.
        """
        if not text:
            return ""

        # Normalize Unicode (removes accents, weird encodings)
        text = unicodedata.normalize("NFKC", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[,*__]", " ", text)
        text = re.sub(r"\n", " ", text)
        # Remove OCR garbage: sequences of non-alphanumerics > 3 chars
        text = re.sub(r"[^\w\s]{3,}", " ", text)

        # Remove obvious OCR artifacts (random capital sequences)
        text = re.sub(r"[A-Z]{4,}\d*", " ", text)

        return text.strip()

    def deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Removes exact duplicate document contents.
        """
        seen = set()
        unique_docs = []
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        logger.info(f"Deduplicated: {len(documents)} → {len(unique_docs)} docs")
        return unique_docs

    def process_local_documents(self) -> List[Document]:
        """
        Loads, cleans, and returns local documents.
        """
        documents = []
        supported_extensions = [
            ".pdf",
            ".txt",
            ".md",
            ".markdown",
            ".docx",
            ".html",
            ".htm",
            ".png",
            ".jpg",
            ".jpeg",
        ]

        for file_path in self.config.DATA_PATH.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                logger.info(f"Processing: {file_path}")
                content = self.load_document(file_path)
                if content.strip():
                    cleaned = self.clean_text(content)
                    if cleaned:  # avoid empty after cleaning
                        documents.append(
                            Document(
                                page_content=cleaned,
                                metadata={
                                    "source": str(file_path),
                                    "file_type": file_path.suffix.lower(),
                                },
                            )
                        )
                else:
                    logger.warning(f"No content extracted from: {file_path}")

        documents = self.deduplicate_documents(documents)
        logger.info(f"Loaded {len(documents)} local documents after cleaning")
        return documents

    def save_to_jsonl(self, documents: List[Document], json_path: Path):
        with open(json_path, "w", encoding="utf-8") as f:
            for doc in documents:
                json.dump(
                    {"text": doc.page_content, "metadata": doc.metadata},
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
        logger.info(f"Saved dataset to {json_path}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document]) -> FAISS:
        if not chunks:
            raise ValueError("No chunks provided for vectorstore creation")
        logger.info("Creating embeddings and vectorstore...")
        return FAISS.from_documents(chunks, self.embeddings)

    def save_vectorstore(self, vectorstore: FAISS):
        self.config.VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.config.VECTORSTORE_PATH))
        logger.info(f"Vectorstore saved to: {self.config.VECTORSTORE_PATH}")

    def ingest_documents(self):
        logger.info("Starting document ingestion...")

        local_docs = self.process_local_documents()
        web_docs = self.load_web_pages()

        all_docs = local_docs + web_docs

        if not all_docs:
            raise ValueError("No documents found to ingest")

        self.save_to_jsonl(all_docs, Path("data/dataset.jsonl"))

        chunks = self.split_documents(all_docs)
        vectorstore = self.create_vectorstore(chunks)
        self.save_vectorstore(vectorstore)

        logger.info("Document ingestion completed successfully!")
        return vectorstore


def main():
    Config.setup_logging()
    ingestor = DocumentIngestor()
    try:
        ingestor.ingest_documents()
        print("✅ Documents ingested successfully!")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"❌ Ingestion failed: {e}")


if __name__ == "__main__":
    main()
