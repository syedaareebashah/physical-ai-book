from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_text_splitter(chunk_size: int = 800, chunk_overlap: int = 200):
    """
    Create a text splitter with specified chunk size and overlap
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks with specified size and overlap
    """
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks


def split_pdf_content(pdf_content: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """
    Split PDF content into chunks (uses the same logic as general text splitting)
    """
    return split_text(pdf_content, chunk_size, chunk_overlap)


def split_markdown_content(markdown_content: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """
    Split Markdown content into chunks (uses the same logic as general text splitting)
    """
    return split_text(markdown_content, chunk_size, chunk_overlap)