import csv
import datetime
import json
import os
import re

import yaml
from bs4 import BeautifulSoup, Comment
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata


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


# --- Custom JSON Encoder ---
class CustomJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder to handle datetime objects by converting them to ISO format strings.
    """

    def default(self, o):
        if isinstance(o, (datetime.datetime, datetime.date)):
            return o.isoformat()
        return super().default(o)


def _parse_mermaid_logic(diagram_text):
    """A dedicated helper function to parse mermaid diagram text robustly."""
    try:
        # 1. Pre-clean the diagram text
        cleaned_text = diagram_text.replace("&gt;", ">")
        cleaned_text = re.sub(r"(?m)^\s*(%%|classDef|class|```).*$", "", cleaned_text)

        # 2. Most robust regex to capture all node content
        nodes = re.findall(
            r'(\w+)\s*\(\s*\["?(.*?)"?\]\s*\);?', cleaned_text, re.DOTALL
        )

        node_map = {}
        for node_id, label in nodes:
            clean_label = re.sub(r"(\\n|<br\s*\/?>)", " ", label).strip()
            node_map[node_id] = clean_label

        # 3. Parse connections
        connections = re.findall(r"(\w+)\s*-->\s*(\w+)", cleaned_text)

        path = []

        if connections:
            # If connections exist, find the path
            sources = {c[0] for c in connections}
            destinations = {c[1] for c in connections}
            start_node = next(iter(sources - destinations), connections[0][0])

            path_map = dict(connections)
            current = start_node
            while current in path_map and current not in path:
                path.append(current)
                current = path_map[current]
            path.append(current)
        elif node_map:
            # If no connections but nodes exist (malformed case), assume alphabetical order
            path = sorted(node_map.keys())

        # 4. Build the final text
        step_list = [
            f"Step {i + 1}: {node_map.get(node_id, '...')}"
            for i, node_id in enumerate(path)
            if node_map.get(node_id)
        ]

        if not step_list:
            return diagram_text  # Return original if parsing yields nothing

        # Find any text that was *before* the diagram
        pre_diagram_text = diagram_text.split(nodes[0][0])[0]
        pre_diagram_text = re.sub(r"```mermaid.*", "", pre_diagram_text).strip()

        return f"{pre_diagram_text}\nThe process is as follows:\n" + "\n".join(
            step_list
        )

    except Exception:
        # On any failure, just strip the styling and return
        return re.sub(r"(?m)^\s*(%%|classDef|class).*$", "", diagram_text)


def convert_diagram_to_text(content):
    # Pass 1: Handle properly wrapped ```mermaid blocks
    processed_content = re.sub(
        r"```mermaid(.*?)```",
        lambda m: _parse_mermaid_logic(m.group(1)),
        content,
        flags=re.DOTALL,
    )

    # Pass 2: Handle "naked" diagram syntax if it's still present
    # Check if it looks like a diagram but wasn't converted in Pass 1
    is_likely_naked_diagram = (
        "-->" in processed_content or "&gt;" in processed_content
    ) and "graph TB" in processed_content

    if is_likely_naked_diagram and "The process is as follows" not in processed_content:
        # It's a naked diagram, so process the whole string
        processed_content = _parse_mermaid_logic(processed_content)

    # Continue to remove plantuml as before
    processed_content = re.sub(
        r"```plantuml.*?```", "", processed_content, flags=re.DOTALL
    )

    return processed_content


def clean_chunk_content(content):
    # Check if the content looks like it contains any part of a mermaid diagram
    if ("-->" in content or "&gt;" in content or "classDef" in content) and re.search(
        r"\w+\(\[", content
    ):
        text = _parse_mermaid_logic(content)
    else:
        text = content

    # Use BeautifulSoup to remove any remaining HTML tags and comments
    soup = BeautifulSoup(text, "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    for tag in soup.find_all(True):
        tag.unwrap()  # Removes tag, keeps content

    text = str(soup)
    # Final cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
            metadata = yaml.safe_load(yaml_str)
            if not isinstance(metadata, dict):
                metadata = {}
        except yaml.YAMLError as e:
            metadata["yaml_parse_warning"] = str(e)
            for line in yaml_str.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip()

        metadata["full_path"] = file_path
        return content_str.strip(), metadata
    else:
        return raw_content.strip(), {"full_path": file_path}


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

        if is_section_heading:
            if current_chunk_paragraphs:
                chunk_content = "\n\n".join(current_chunk_paragraphs)
                if len(chunk_content) >= min_chunk_size:
                    chunks.append(
                        Document(page_content=chunk_content, metadata=metadata)
                    )
            section_header = paragraph
            subsection_header = ""
            current_chunk_paragraphs = [section_header]
            continue

        if is_subsection_heading:
            subsection_header = paragraph

        current_content = "\n\n".join(current_chunk_paragraphs)
        if (
            len(current_content) + len(paragraph) > max_chunk_size
            and len(current_content) >= min_chunk_size
        ):
            chunks.append(Document(page_content=current_content, metadata=metadata))
            current_chunk_paragraphs = []
            if section_header:
                current_chunk_paragraphs.append(section_header)
            if subsection_header:
                current_chunk_paragraphs.append(subsection_header)

        current_chunk_paragraphs.append(paragraph)

    if current_chunk_paragraphs:
        final_content = "\n\n".join(current_chunk_paragraphs)
        if len(final_content) >= min_chunk_size:
            chunks.append(Document(page_content=final_content, metadata=metadata))

    if not chunks:
        chunks.append(Document(page_content=content, metadata=metadata))

    return chunks


# --- Step 3: The "Smart Dispatcher" ---
def chunk_document(file_path, root_directory):  # <-- Pass root_directory
    """
    Loads a document, enriches metadata, and routes it to the best splitting strategy.
    """
    print(f"\nProcessing file: {file_path}")
    content, metadata = load_and_preprocess_file(file_path)
    if content is None:
        return []

    # --- NEW: Enrich Metadata based on file path ---
    relative_path = os.path.relpath(file_path, root_directory)

    # 1. origin
    metadata["origin"] = "dswiki"

    # 2. source (the relative path is more useful)
    metadata["source"] = relative_path.replace(
        "\\", "/"
    )  # Use forward slashes for consistency

    # 3. section
    path_parts = relative_path.split(os.sep)
    if len(path_parts) > 1:
        section_name = path_parts[0]
        # Format the section name nicely (e.g., 'use_cases' -> 'Use Cases')
        metadata["section"] = section_name.replace("_", " ").replace("-", " ").title()
    else:
        metadata["section"] = "General"  # Fallback for files in the root

    # 4. id
    path_without_ext, _ = os.path.splitext(relative_path)
    id_path = path_without_ext.replace(os.sep, "-").lower()
    metadata["id"] = f"dswiki-{id_path}"
    # --- End of Metadata Enrichment ---

    # Get chunks using the appropriate strategy
    if "faq" in file_path.lower():
        print("-> Strategy: FAQ/HTML Details Splitting")
        chunks = split_faq_style(content, metadata)
    else:
        print("-> Strategy: Context-Aware Structural Splitting")
        chunks = split_markdown_by_structure(content, metadata)

    # --- NEW: Clean the content of each chunk ---
    cleaned_chunks = []
    for chunk in chunks:
        cleaned_content = clean_chunk_content(chunk.page_content)
        # We only keep chunks that still have meaningful content after cleaning
        if (
            cleaned_content and len(cleaned_content) > 50
        ):  # Avoid empty or very short chunks
            chunk.page_content = cleaned_content
            cleaned_chunks.append(chunk)

    return cleaned_chunks


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
                # Pass root_directory to the chunker for context
                chunks = chunk_document(file_path, root_directory)
                all_chunks.extend(chunks)

    return all_chunks


# Function to Generate Embeddings and Save to ChromaDB
def create_and_persist_vectordb(chunks, persist_directory):
    if not chunks:
        print("No chunks to process. Vector database will not be created.")
        return

    # --- FIX: Filter metadata to remove complex types before sending to Chroma ---
    # This will convert datetime objects to strings and remove other unsupported types.
    filtered_chunks = filter_complex_metadata(chunks)

    print("\nInitializing embedding model...")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    print("-> Model: nomic-embed-text")

    print(f"\nCreating and persisting vector database to '{persist_directory}'...")
    print(f"This may take a while, embedding {len(filtered_chunks)} chunks...")

    Chroma.from_documents(
        documents=filtered_chunks,  # <-- Use the filtered chunks
        embedding=embedding_model,
        persist_directory=persist_directory,
    )

    print("-> Vector database created and saved successfully.")
    print(f"-> You can now load it from '{persist_directory}' in other applications.")


def export_chunks(chunks, output_dir="."):
    """
    Exports the list of Document chunks to JSON, CSV, and JSONL files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_to_export = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]

    # --- Export to JSON ---
    json_path = os.path.join(output_dir, "chunks_datastore.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            data_to_export, f, indent=4, ensure_ascii=False, cls=CustomJSONEncoder
        )
    print(f"\nSuccessfully exported {len(chunks)} chunks to {json_path}")

    # --- Export to CSV ---
    csv_path = os.path.join(output_dir, "chunks_datastore.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        all_meta_keys = set()
        for item in data_to_export:
            all_meta_keys.update(item["metadata"].keys())

        # Define a preferred order for the main columns
        preferred_order = ["id", "origin", "section", "source", "page_content"]
        # Add the rest of the metadata keys alphabetically
        remaining_keys = sorted(list(all_meta_keys - set(preferred_order)))
        fieldnames = preferred_order + remaining_keys

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in data_to_export:
            row = {"page_content": item["page_content"]}
            for key, value in item["metadata"].items():
                if isinstance(value, (datetime.datetime, datetime.date)):
                    row[key] = value.isoformat()
                else:
                    row[key] = value
            writer.writerow(row)
    print(f"Successfully exported {len(chunks)} chunks to {csv_path}")

    # --- Export to JSONL ---
    jsonl_path = os.path.join(output_dir, "chunks_datastore.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data_to_export:
            f.write(json.dumps(item, ensure_ascii=False, cls=CustomJSONEncoder) + "\n")
    print(f"Successfully exported {len(chunks)} chunks to {jsonl_path}")


# Main processing function to be called from other scripts
def run_dswiki_processing(root_directory, output_directory, chroma_persist_directory):
    """
    Executes the full processing pipeline for the DSWiki data source.
    """
    print("--- Starting DSWiki Processing ---")

    # Step 1: Process all markdown files into chunks
    final_chunks = process_all_markdown_files(root_directory)

    if final_chunks:
        # Step 2 (Optional): Export chunks to files for inspection
        export_chunks(final_chunks, output_directory)

        # Step 3: Generate embeddings and save them to ChromaDB
        create_and_persist_vectordb(final_chunks, chroma_persist_directory)

        print(f"--- DSWiki Processing Complete. Total Chunks: {len(final_chunks)} ---")
    else:
        print("No DSWiki markdown files were found or processed.")


if __name__ == "__main__":
    # Default paths for standalone execution
    ROOT_DIRECTORY = "./data/raw/wikijs"
    OUTPUT_DIRECTORY = "./data/processed/wikijs"
    CHROMA_PERSIST_DIRECTORY = "./desi_vectordb"

    run_dswiki_processing(ROOT_DIRECTORY, OUTPUT_DIRECTORY, CHROMA_PERSIST_DIRECTORY)
