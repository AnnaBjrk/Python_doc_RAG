"""Advanced chunker for Python documentation in text format.
This module processes Python documentation files, extracts relevant sections,
and creates a vector store for efficient retrieval.
It handles various content types such as code examples, tables, and sections,
and builds a structured representation of the documentation.
chunking is done using LangChain's text splitter and embeddings.
It also supports metadata extraction for better context in retrieval.
chunking logic:
- Main sections are identified by titles followed by long separators (=== or ***).
- Sub-sections are identified by shorter separators (===, ---, ~~~).
-If a section is too large, it is split by function definitions.
- Special structures like code blocks and tables are preserved.
        content_structures = {
            "code_blocks": [],
            "tables": [],
            "lists": [] - not added in this version might be done
- Code blocks and tables are preserved as special structures.
#
- Content is chunked based on size limits, with overlap for context.
"""

import os
import re
import glob
from typing import List, Dict, Tuple, Any, Optional
# import json
from dataclasses import dataclass, field, asdict
# import math
from pathlib import Path

# import for whoosh - trad search engine
from whoosh.fields import Schema, TEXT, ID, KEYWORD
from whoosh.analysis import KeywordAnalyzer
from whoosh import index
from whoosh.qparser import QueryParser


# LangChain imports

from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangChainDocument


@dataclass
class DocMetadata:
    """Metadata for a Python documentation chunk."""
    file_path: str
    module_name: str
    title: str
    section_path: List[str] = field(default_factory=list)
    is_code_example: bool = False
    is_table: bool = False
    related_modules: List[str] = field(default_factory=list)
    python_version: str = "3.13"  # Default based on your example
    # explanation, api_reference, tutorial, etc.
    content_type: str = "explanation"
    parent_chunk_id: Optional[str] = None
    chunk_type: str = "content"  # content, section_header, module_index, etc.
    function_name: Optional[str] = None  # Added for function metadata


# Add this simple search class
class WhooshSearch:
    """Simple wrapper for Whoosh keyword search."""

    def __init__(self, index_dir="./my_whoosh_index"):
        self.index_dir = index_dir
        self.ix = index.open_dir(index_dir)
        self.parser = QueryParser("content", self.ix.schema)

    def search(self, query_text, limit=10):
        """Search for exact keywords and return results."""
        with self.ix.searcher() as searcher:
            query = self.parser.parse(query_text)
            results = searcher.search(query, limit=limit)

            search_results = []
            for result in results:
                search_results.append({
                    'content': result['content'],
                    'document': result['document'],
                    'folder': result['folder'],
                    'section_path': result['section_path'],
                    'function_name': result['function_name'],
                    'score': result.score
                })

            return search_results


class PythonDocChunker:
    """
    Advanced chunker for Python documentation in text format.
    """

    def __init__(
        self,
        docs_dir: str,
        chunk_min_size: int = 50,
        chunk_max_size: int = 4000,
        chunk_overlap: int = 200,
        embeddings=None
    ):
        self.docs_dir = Path(docs_dir)
        self.output_dir = "./my_vectorstore"
        self.chunk_min_size = chunk_min_size
        self.chunk_max_size = chunk_max_size
        self.chunk_overlap = chunk_overlap
        self.module_categories = self._build_module_categories()
        self.module_relationships = {}
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5")

        # Regex patterns - Updated to handle os.txt format better
        # Main sections: title with *** or major separator with long ====
        self.main_section_pattern = re.compile(
            r'([^\n]+)\n(?:[*]{3,}|[=]{50,})\n')
        # Sub-sections: headings with medium === (less than 50 chars) or --- or ~~~
        self.subsection_pattern = re.compile(
            r'([^\n]+)\n(?:[=]{10,49}|[-~]{3,})\n')
        self.code_block_pattern = re.compile(
            r'(>>> .*?(?:\n|$)(?:... .*?(?:\n|$))*)', re.DOTALL)
        self.table_pattern = re.compile(
            r'(\+[-+]+\+\n(?:[|].*?[|]\n)+\+[-+]+\+)', re.DOTALL)
        # Updated pattern to match Python documentation format: sys.functionname(params) or module.functionname(params)
        # Also handles cases where function definitions start at beginning of line
        self.function_pattern = re.compile(
            r'(?:^|\n)\s*(?:[a-zA-Z_][a-zA-Z0-9_.]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:\n|$)', re.MULTILINE)

        # Text splitter for content-based chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_max_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _build_module_categories(self) -> Dict[str, str]:
        """
        Build a mapping of modules to their categories.
        Ideally this would parse the index.txt or contents.txt to get real categories.
        For now, we'll use a simplified approach based on directory structure.
        """
        categories = {}

        # Map top-level directories to categories
        directory_categories = {
            "library": "Standard Library",
            "c-api": "C API",
            "reference": "Language Reference",
            "tutorial": "Tutorials",
            "howto": "How-To Guides",
            "faq": "FAQs",
            "extending": "Extending Python",
            "distributing": "Distribution",
            "whatsnew": "What's New",
            "about_the_documentation": "About the Documentation",
            "installing": "Installation Guide",
            "using": "Using Python"
        }

        # Scan all txt files and assign categories
        for file_path in glob.glob(str(self.docs_dir / "**/*.txt"), recursive=True):
            rel_path = os.path.relpath(file_path, self.docs_dir)
            parts = Path(rel_path).parts

            if len(parts) > 0:
                top_dir = parts[0]  # förtsta foldern i sökvägen
                module_name = Path(rel_path).stem  # filnamn utan .txt

                # returnear "Uncategorized" om den inte finns i mappen
                category = directory_categories.get(top_dir, "Uncategorized")
                categories[module_name] = category

                # For library modules, a large catalog, we refine categories further based on actual Python categories, The reference to
                # the standard library is lost
                if top_dir == "library" and len(parts) > 1:
                    # Example: add subcategories based on directory structure or known module groups
                    if module_name in ["asyncio", "threading", "multiprocessing"]:
                        categories[module_name] = "Concurrency"
                    elif module_name in ["re", "string", "difflib", "textwrap"]:
                        categories[module_name] = "Text Processing"
                    elif module_name in ["http", "urllib", "ftplib", "smtplib", "xmlrpc"]:
                        categories[module_name] = "Internet & Web"
                    elif module_name in ["json", "csv", "pickle", "sqlite3", "dbm"]:
                        categories[module_name] = "Data Formats"
                    # ... and so on

        return categories

    def _extract_module_relationships(self):  # To do
        """
        Extract relationships between modules by scanning for cross-references.
        This would ideally build a graph of module relationships.
        """
        # Placeholder for a more sophisticated implementation
        # Would scan all content for module references and build a relationship graph
        pass

    def _extract_title(self, content: str) -> str:
        """Extract the title from content."""
        title_match = re.search(r'^(.*?)\n[=*]+', content)
        return title_match.group(1) if title_match else ""

    def _contains_table(self, content: str) -> bool:
        """Check if the content contains a table."""
        return bool(self.table_pattern.search(content))

    def _contains_code(self, content: str) -> bool:
        """Check if the content contains code examples."""
        return ">>>" in content or "```" in content or "    " in content

    def _detect_related_modules(self, content: str) -> List[str]:
        """
        Detect mentions of other Python modules in the content.
        Based on analysis of Python 3.13 documentation showing libraries with 10+ occurrences.
        """
        # Libraries with 10+ occurrences in Python documentation (sorted by frequency)
        standard_libs = [
            "logging", "sys", "time", "datetime", "os",
            "enum", "__future__", "asyncio", "typing", "multiprocessing",
            "argparse", "urllib", "decimal", "contextlib", "math",
            "random", "socket", "tkinter", "collections", "ctypes",
            "functools", "json", "array", "io", "re", "threading",
            "csv", "hashlib", "http", "importlib", "unittest",
            "turtle", "fractions", "configparser", "pprint", "string",
            "pdb", "shutil", "smtplib", "struct"
        ]

        related = []
        for lib in standard_libs:
            # Look for import statements or module references
            if f"import {lib}" in content or f"from {lib}" in content or f"{lib}." in content:
                related.append(lib)

        return related

    def _preserve_special_structures(self, content: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Identify and preserve special structures like code blocks and tables.
        Returns a list of (content, metadata) tuples.
        """
        preserved_chunks = []

        # Preserve code blocks
        code_blocks = self.code_block_pattern.findall(content)
        for block in code_blocks:
            metadata = {
                "is_code_example": True,
                "content_type": "example"
            }
            preserved_chunks.append((block, metadata))

        # Preserve tables
        tables = self.table_pattern.findall(content)
        for table in tables:
            metadata = {
                "is_table": True,
                "content_type": "reference"
            }
            preserved_chunks.append((table, metadata))

        return preserved_chunks

    def _chunk_by_functions(self, content: str, base_metadata: DocMetadata) -> List[Tuple[str, DocMetadata]]:
        """
        Split content by function definitions when content exceeds max size.
        Returns a list of (content, metadata) tuples.
        """
        chunks = []

        # Find all function matches
        function_matches = list(self.function_pattern.finditer(content))

        if not function_matches:
            return []  # No functions found

        # Split content by function positions
        last_end = 0

        for i, match in enumerate(function_matches):
            function_name = match.group(1)
            start_pos = match.start()

            # Determine end position (start of next function or end of content)
            if i + 1 < len(function_matches):
                end_pos = function_matches[i + 1].start()
            else:
                end_pos = len(content)

            # Extract function content including any preceding content
            if i == 0 and start_pos > 0:
                # Include any content before the first function
                preceding_content = content[last_end:start_pos].strip()
                if preceding_content:
                    metadata = DocMetadata(**asdict(base_metadata))
                    chunks.append((preceding_content, metadata))

            # Extract function content
            function_content = content[start_pos:end_pos].strip()
            if function_content:
                metadata = DocMetadata(**asdict(base_metadata))
                metadata.function_name = function_name
                metadata.content_type = "function"
                chunks.append((function_content, metadata))

        return chunks

    def _chunk_section(
        self,
        content: str,
        base_metadata: DocMetadata,
        section_title: str = None
    ) -> List[Tuple[str, DocMetadata]]:
        """
        Chunk a section of content, respecting special structures.
        Returns a list of (content, metadata) tuples.
        """
        chunks = []

        # Update metadata with section info if provided
        if section_title:
            section_path = base_metadata.section_path.copy()
            section_path.append(section_title)
            base_metadata.section_path = section_path

        # Handle special structures
        special_structures = self._preserve_special_structures(content)
        if special_structures:
            # For each special structure, create a separate chunk
            for special_content, special_meta in special_structures:
                # Create a copy of base metadata and update it
                metadata = DocMetadata(**asdict(base_metadata))
                metadata.is_code_example = special_meta.get(
                    "is_code_example", False)
                metadata.is_table = special_meta.get("is_table", False)
                metadata.content_type = special_meta.get(
                    "content_type", metadata.content_type)

                chunks.append((special_content, metadata))

                # Remove the special content from the main content to avoid duplication
                content = content.replace(special_content, "")

        # If we still have content after removing special structures, chunk it
        if content.strip():

            # Use LangChain's text splitter for the remaining content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_max_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )

            texts = text_splitter.split_text(content)

            for i, text in enumerate(texts):
                # Create a copy of metadata for each chunk
                metadata = DocMetadata(**asdict(base_metadata))

                # Update with related modules if we can detect them
                metadata.related_modules = self._detect_related_modules(text)

                chunks.append((text, metadata))

        return chunks

    def process_file(self, file_path: str) -> List[Tuple[str, DocMetadata]]:
        """
        Process a single documentation file and return chunks with metadata.
        """
        rel_path = os.path.relpath(file_path, self.docs_dir)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract basic metadata
        module_name = Path(rel_path).stem
        title = self._extract_title(content)

        # Create base metadata
        base_metadata = DocMetadata(
            file_path=rel_path,
            module_name=module_name,
            title=title,
            related_modules=[],  # Will be populated per chunk
            python_version="3.13",  # Could extract from path or content
        )

        chunks = []

        # Split by main sections first
        sections = self.main_section_pattern.split(content)
        if len(sections) > 1:
            # First part is the module intro
            intro_text = sections[0].strip()
            if intro_text:
                # Create module overview chunk
                intro_metadata = DocMetadata(**asdict(base_metadata))
                intro_metadata.chunk_type = "module_overview"
                chunks.append((intro_text, intro_metadata))

            # Process each main section
            for i in range(1, len(sections), 2):
                section_title = sections[i].strip()
                section_content = sections[i+1] if i+1 < len(sections) else ""

                if not section_content.strip():
                    continue

                # Update section path
                section_metadata = DocMetadata(**asdict(base_metadata))
                section_metadata.section_path = [section_title]

                # Check if section should be further split into subsections
                subsections = self.subsection_pattern.split(section_content)

                if len(subsections) > 1:
                    # First part may contain section intro
                    section_intro = subsections[0].strip()
                    if section_intro:
                        chunks.extend(self._chunk_section(
                            section_intro, section_metadata))

                    # Process each subsection
                    for j in range(1, len(subsections), 2):
                        subsection_title = subsections[j].strip()
                        subsection_content = subsections[j +
                                                         1] if j+1 < len(subsections) else ""

                        if not subsection_content.strip():
                            continue

                        # Create subsection metadata
                        subsection_metadata = DocMetadata(
                            **asdict(section_metadata))
                        subsection_metadata.section_path.append(
                            subsection_title)

                        # Chunk the subsection
                        chunks.extend(self._chunk_section(
                            subsection_content, subsection_metadata))
                else:
                    # No subsections, check if section is too large and contains functions
                    if len(section_content) > self.chunk_max_size:
                        # Try to split by functions first
                        function_chunks = self._chunk_by_functions(
                            section_content, section_metadata)
                        if function_chunks:
                            chunks.extend(function_chunks)
                        else:
                            # No functions found, fall back to regular chunking
                            chunks.extend(self._chunk_section(
                                section_content, section_metadata))
                    else:
                        # Section is small enough, chunk normally
                        chunks.extend(self._chunk_section(
                            section_content, section_metadata))
        else:
            # No clear sections, chunk the whole file
            chunks.extend(self._chunk_section(content, base_metadata))

        return chunks

    def process_all_files(self) -> List[LangChainDocument]:
        """
        Process all documentation files and return LangChain Document objects.
        """
        all_docs = []

        # Find all .txt files in the docs directory
        for file_path in glob.glob(str(self.docs_dir / "**/*.txt"), recursive=True):
            try:
                chunks = self.process_file(file_path)

                # Convert to LangChain documents
                for text, metadata in chunks:
                    doc = LangChainDocument(
                        page_content=text,
                        metadata=asdict(metadata)
                    )
                    all_docs.append(doc)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        # Create module index documents that help with navigation
        self._create_module_indexes(all_docs)

        return all_docs

    def _create_module_indexes(self, documents: List[LangChainDocument]):
        """
        Create special index documents that map the structure of the documentation.
        These are helpful for providing context about document organization.
        """
        # Group documents by module
        module_docs = {}
        for doc in documents:
            module = doc.metadata.get("module_name")
            if module not in module_docs:
                module_docs[module] = []
            module_docs[module].append(doc)

        # For each module, create an index document
        for module, docs in module_docs.items():
            # Extract all section paths
            sections = set()
            for doc in docs:
                section_path = doc.metadata.get("section_path", [])
                if isinstance(section_path, str):
                    if section_path:
                        sections.add(section_path)
                else:
                    # Handle if section_path is still a list
                    for i in range(len(section_path)):
                        sections.add(" > ".join(section_path[:i+1]))

            # Create a hierarchical representation
            sections_text = "\n".join(sorted(list(sections)))

            # Find a representative doc to get basic metadata
            if docs:
                rep_doc = docs[0]
                index_content = f"MODULE: {module}\n"
                index_content += f"TITLE: {rep_doc.metadata.get('title', module)}\n"
                index_content += "\nSECTIONS:\n" + sections_text

                # Create metadata for the index
                index_metadata = {
                    "file_path": rep_doc.metadata.get("file_path"),
                    "module_name": module,
                    "title": rep_doc.metadata.get("title", module),
                    "section_path": "",
                    "is_code_example": False,
                    "is_table": False,
                    "related_modules": "",  # Could add related modules here
                    "python_version": rep_doc.metadata.get("python_version", "3.13"),
                    "content_type": "module_index",
                    "chunk_type": "module_index",
                    "function_name": None
                }

                # Add the index document
                index_doc = LangChainDocument(
                    page_content=index_content,
                    metadata=index_metadata
                )
                documents.append(index_doc)

    def build_vectorstore(self, documents: List[LangChainDocument], persist_directory: str = None):
        """
        Build a vector store from the chunked documents.
        """
        # Convert list values to strings in metadata before creating vectorstore
        for doc in documents:
            # Convert section_path list to string with delimiter
            if "section_path" in doc.metadata and isinstance(doc.metadata["section_path"], list):
                doc.metadata["section_path"] = " > ".join(
                    doc.metadata["section_path"])

            # Convert related_modules list to string with delimiter
            if "related_modules" in doc.metadata and isinstance(doc.metadata["related_modules"], list):
                doc.metadata["related_modules"] = ", ".join(
                    doc.metadata["related_modules"])

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )

        if persist_directory:
            vectorstore.persist()

        return vectorstore

    def chunk_and_create_vector_database(self, create_whoosh=True):
        """
        Process Python documentation and create a vector database and Whoosh index.

        Args:
            docs_dir (str): Path to the Python documentation directory
            output_dir (str): Output directory for vector store
            chunk_size (int): Maximum chunk size
            chunk_overlap (int): Chunk overlap
            embedding_model: The embedding model to use (defaults to HuggingFaceEmbeddings)

        Returns:
            The created vector store
        """

        # Process all files
        print("Processing documentation files...")
        documents = self.process_all_files()
        print(f"Created {len(documents)} document chunks")

        # Build vector store
        print(f"Building vector store at {self.output_dir}...")
        vectorstore = self.build_vectorstore(documents, self.output_dir)

        # Build Whoosh index if requested
        whoosh_search = None
        if create_whoosh:
            try:
                print("Building Whoosh keyword index...")
                self.build_whoosh_index(documents)  # Creates the index files
                whoosh_search = WhooshSearch(
                    "./my_whoosh_index")  # Wraps it properly
                print("Whoosh index created successfully")
            except Exception as e:
                print(f"Warning: Failed to create Whoosh index: {e}")
                print("Continuing with vector search only...")

        print("Done!")
        return vectorstore, whoosh_search

    def _create_whoosh_schema(self):
        """Create Whoosh schema for keyword search."""
        return Schema(
            doc_id=ID(stored=True, unique=True),
            folder=KEYWORD(stored=True),
            document=KEYWORD(stored=True),
            section_path=TEXT(stored=True),
            content=TEXT(analyzer=KeywordAnalyzer(), stored=True),
            function_name=KEYWORD(stored=True)
        )

    def build_whoosh_index(self, documents, whoosh_dir="./my_whoosh_index"):
        """Build Whoosh index from the same documents used for vector store."""

        # Create index directory
        os.makedirs(whoosh_dir, exist_ok=True)

        # Create or open index
        schema = self._create_whoosh_schema()
        if index.exists_in(whoosh_dir):
            ix = index.open_dir(whoosh_dir)
        else:
            ix = index.create_in(whoosh_dir, schema)

        # Add documents to index
        writer = ix.writer()

        for i, doc in enumerate(documents):
            writer.add_document(
                doc_id=str(i),
                folder=doc.metadata.get('file_path', '').split(
                    '/')[0] if '/' in doc.metadata.get('file_path', '') else '',
                document=doc.metadata.get('module_name', ''),
                section_path=doc.metadata.get('section_path', ''),
                content=doc.page_content,
                function_name=doc.metadata.get('function_name', '') or ''
            )

        writer.commit()
        print(f"Whoosh index created with {len(documents)} documents")
        return ix
