import os
import re

import yaml
from bs4 import BeautifulSoup


# --- Helper Class for Storing Chunks ---
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        metadata_str = ", ".join(f"{k}='{v}'" for k, v in self.metadata.items())
        content_preview = self.page_content[:200].strip().replace("\n", " ")
        return (
            f"--- CHUNK ---\n"
            f"Metadata: {metadata_str}\n"
            f"Content: '{content_preview}...'\n"
            f"-------------\n"
        )


# --- Step 1: ROBUST Pre-processing and Loading ---
def load_and_preprocess_file(file_path):
    """
    Loads a file, separates YAML frontmatter, and handles potential parsing errors.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            raw_content = f.read()
    except FileNotFoundError:
        return None, None

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", raw_content, re.DOTALL)

    if match:
        yaml_str, content_str = match.groups()
        metadata = {}
        try:
            # Try to parse the YAML strictly
            metadata = yaml.safe_load(yaml_str)
            if not isinstance(metadata, dict):
                metadata = {}
        except yaml.YAMLError as e:
            # If parsing fails, fall back to simple line-by-line regex
            metadata["yaml_parse_warning"] = str(e)
            for line in yaml_str.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip()

        metadata["source"] = file_path
        return content_str.strip(), metadata
    else:
        return raw_content.strip(), {"source": file_path}


# --- Step 2: The Splitting Strategies ---


def split_faq_style(content, metadata):
    """Strategy A: Specialized splitter for FAQ files using <details> tags."""
    soup = BeautifulSoup(content, "html.parser")
    details_tags = soup.find_all("details")
    chunks = []
    if not details_tags:
        return [Document(page_content=content, metadata=metadata)]
    for tag in details_tags:
        summary = tag.find("summary")
        question = summary.get_text(strip=True) if summary else "No Question"
        answer = "\n".join([s for s in tag.stripped_strings if s != question])
        chunk_content = f"Question: {question}\nAnswer: {answer}"
        chunk_metadata = metadata.copy()
        chunk_metadata["faq_question"] = question
        chunks.append(Document(page_content=chunk_content, metadata=chunk_metadata))
    return chunks


def split_markdown_by_structure(
    content, metadata, min_chunk_size=200, max_chunk_size=1500
):
    """
    Strategy B: Context-aware chunking based on the user's provided script logic.
    It preserves header context in each chunk.
    """
    paragraphs = [p for p in content.split("\n\n") if p.strip()]
    chunks = []
    current_chunk_paragraphs = []
    section_header = ""
    subsection_header = ""

    for paragraph in paragraphs:
        is_section_heading = paragraph.startswith("## ")
        is_subsection_heading = paragraph.startswith("### ")

        # If we start a new major section, save the previous chunk
        if is_section_heading:
            if current_chunk_paragraphs:
                chunk_content = "\n\n".join(current_chunk_paragraphs)
                if len(chunk_content) >= min_chunk_size:
                    chunks.append(
                        Document(page_content=chunk_content, metadata=metadata)
                    )

            # Reset and start a new chunk context
            section_header = paragraph
            subsection_header = ""
            current_chunk_paragraphs = [section_header]
            continue

        if is_subsection_heading:
            subsection_header = paragraph

        # Check if adding the next paragraph would exceed the max size
        current_content = "\n\n".join(current_chunk_paragraphs)
        if (
            len(current_content) + len(paragraph) > max_chunk_size
            and len(current_content) >= min_chunk_size
        ):
            chunks.append(Document(page_content=current_content, metadata=metadata))

            # Start a new chunk, preserving the header context
            current_chunk_paragraphs = []
            if section_header:
                current_chunk_paragraphs.append(section_header)
            if subsection_header:
                current_chunk_paragraphs.append(subsection_header)

        current_chunk_paragraphs.append(paragraph)

    # Add the final chunk
    if current_chunk_paragraphs:
        final_content = "\n\n".join(current_chunk_paragraphs)
        if len(final_content) >= min_chunk_size:
            chunks.append(Document(page_content=final_content, metadata=metadata))

    # If no chunks were created (for very short files), chunk the whole content
    if not chunks:
        chunks.append(Document(page_content=content, metadata=metadata))

    return chunks


# --- Step 3: The "Smart Dispatcher" ---
def chunk_document(file_path):
    """
    Loads a document and routes it to the best splitting strategy.
    """
    print(f"\nProcessing file: {file_path}")
    content, metadata = load_and_preprocess_file(file_path)
    if content is None:
        return []

    # Heuristic 1: Use the specialized FAQ splitter for FAQ files
    if "faq" in file_path.lower():
        print("-> Strategy: FAQ/HTML Details Splitting")
        return split_faq_style(content, metadata)

    # Heuristic 2: Use the context-aware structural splitter for all other files
    else:
        print("-> Strategy: Context-Aware Structural Splitting")
        return split_markdown_by_structure(content, metadata)


# --- Main Execution Function ---
def process_all_markdown_files(root_directory):
    """
    Recursively finds and processes all markdown files in a directory.
    """
    if not os.path.isdir(root_directory):
        print(f"Error: The specified directory does not exist: {root_directory}")
        return []

    all_chunks = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith(".md"):
                file_path = os.path.join(dirpath, filename)
                chunks = chunk_document(file_path)
                all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # --- CONFIGURATION: SET YOUR ROOT DIRECTORY HERE ---
    # ----------------------------------------------------------------------
    # Modify this line to point to the folder containing your .md files.
    # For example: "./data/daily_dump" or "C:/Users/YourUser/MyProject/wiki_files"
    YOUR_ROOT_DIRECTORY = "./data/raw/wikijs/daily"
    # ----------------------------------------------------------------------

    # Process the files and get the list of all document chunks
    final_chunks = process_all_markdown_files(YOUR_ROOT_DIRECTORY)

    # Print a summary
    print("\n\n=========================================")
    print(f"      Total Chunks Created: {len(final_chunks)}")
    print("=========================================\n")

    # Print all the resulting chunks
    i = 0
    if final_chunks:
        for chunk in final_chunks:
            if i < 50:
                i += 1
                print(chunk)
            else:
                break
    else:
        print("No markdown files were found or processed.")
