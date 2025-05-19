import os
import re
import glob
from typing import List, Dict, Tuple, Any, Optional, Set
import json
from dataclasses import dataclass, field, asdict
import math
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document as LangChainDocument


@dataclass
class DocMetadata:
    """Metadata for a Python documentation chunk."""
    file_path: str
    module_name: str
    title: str
    section_path: List[str] = field(default_factory=list)
    deprecated: Optional[str] = None
    is_code_example: bool = False
    is_table: bool = False
    related_modules: List[str] = field(default_factory=list)
    python_version: str = "3.13"  # Default based on your example
    content_type: str = "explanation"  # explanation, api_reference, tutorial, etc.
    parent_chunk_id: Optional[str] = None
    chunk_type: str = "content"  # content, section_header, module_index, etc.


class PythonDocChunker:
    """
    Advanced chunker for Python documentation in text format.
    """
    def __init__(
        self, 
        docs_dir: str, 
        chunk_min_size: int = 150,
        chunk_max_size: int = 1000, 
        chunk_overlap: int = 200,
        embeddings = None
    ):
        self.docs_dir = Path(docs_dir)
        self.chunk_min_size = chunk_min_size
        self.chunk_max_size = chunk_max_size
        self.chunk_overlap = chunk_overlap
        self.module_categories = self._build_module_categories()
        self.module_relationships = {}
        self.embeddings = embeddings or OpenAIEmbeddings()
        
        # Regex patterns
        self.main_section_pattern = re.compile(r'([^\n]+)\n[=*]+\n')
        self.subsection_pattern = re.compile(r'([^\n]+)\n[-~]+\n')
        self.code_block_pattern = re.compile(r'(>>> .*?(?:\n|$)(?:... .*?(?:\n|$))*)', re.DOTALL)
        self.table_pattern = re.compile(r'(\+[-+]+\+\n(?:[|].*?[|]\n)+\+[-+]+\+)', re.DOTALL)
        
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
            "whatsnew": "What's New"
        }
        
        # Scan all txt files and assign categories
        for file_path in glob.glob(str(self.docs_dir / "**/*.txt"), recursive=True):
            rel_path = os.path.relpath(file_path, self.docs_dir)
            parts = Path(rel_path).parts
            
            if len(parts) > 0:
                top_dir = parts[0]
                module_name = Path(rel_path).stem
                
                category = directory_categories.get(top_dir, "Uncategorized")
                categories[module_name] = category
                
                # For library modules, we could refine categories further based on actual Python categories
                if top_dir == "library" and len(parts) > 1:
                    # Example: add subcategories based on directory structure or known module groups
                    if module_name in ["asyncio", "threading", "multiprocessing"]:
                        categories[module_name] = "Concurrency"
                    elif module_name in ["re", "string", "difflib"]:
                        categories[module_name] = "Text Processing"
        
        return categories
    
    def _extract_module_relationships(self):
        """
        Extract relationships between modules by scanning for cross-references.
        This would ideally build a graph of module relationships.
        """
        # Placeholder for a more sophisticated implementation
        # Would scan all content for module references and build a relationship graph
        pass
    
    def _extract_title_and_status(self, content: str) -> Tuple[str, Optional[str]]:
        """Extract the title and deprecation status from content."""
        title_match = re.search(r'^(.*?)\n[=*]+', content)
        title = title_match.group(1) if title_match else ""
        
        # Check for deprecation
        deprecated_match = re.search(r'Deprecated since version (\d+\.\d+)', content)
        deprecated = deprecated_match.group(0) if deprecated_match else None
        
        return title, deprecated
    
    def _identify_content_type(self, content: str, file_path: str) -> str:
        """Identify the type of content in the chunk."""
        if "class " in content or "def " in content or re.search(r':[a-zA-Z]+:', content):
            return "api_reference"
        elif "example" in content.lower() or ">>>" in content:
            return "example"
        elif file_path.startswith("tutorial") or file_path.startswith("howto"):
            return "tutorial"
        else:
            return "explanation"
    
    def _contains_table(self, content: str) -> bool:
        """Check if the content contains a table."""
        return bool(self.table_pattern.search(content))
    
    def _contains_code(self, content: str) -> bool:
        """Check if the content contains code examples."""
        return ">>>" in content or "```" in content or "    " in content
    
    def _detect_related_modules(self, content: str) -> List[str]:
        """
        Detect mentions of other Python modules in the content.
        This is simplified - a full implementation would handle more patterns.
        """
        standard_libs = ["os", "sys", "re", "math", "datetime", "collections", 
                         "json", "csv", "pathlib", "asyncio", "threading"]
        
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
                metadata.is_code_example = special_meta.get("is_code_example", False)
                metadata.is_table = special_meta.get("is_table", False)
                metadata.content_type = special_meta.get("content_type", metadata.content_type)
                
                chunks.append((special_content, metadata))
                
                # Remove the special content from the main content to avoid duplication
                content = content.replace(special_content, "")
        
        # If we still have content after removing special structures, chunk it
        if content.strip():
            # Determine optimal chunk size based on content
            content_type = self._identify_content_type(content, base_metadata.file_path)
            base_metadata.content_type = content_type
            
            # Adjust chunk size based on content type
            chunk_size = self.chunk_max_size
            if content_type == "api_reference":
                chunk_size = self.chunk_max_size * 1.5  # Larger chunks for API references
            elif content_type == "example":
                chunk_size = self.chunk_max_size * 2  # Even larger for examples
            
            # Use LangChain's text splitter for the remaining content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
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
        title, deprecated = self._extract_title_and_status(content)
        
        # Create base metadata
        base_metadata = DocMetadata(
            file_path=rel_path,
            module_name=module_name,
            title=title,
            deprecated=deprecated,
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
                        chunks.extend(self._chunk_section(section_intro, section_metadata))
                    
                    # Process each subsection
                    for j in range(1, len(subsections), 2):
                        subsection_title = subsections[j].strip()
                        subsection_content = subsections[j+1] if j+1 < len(subsections) else ""
                        
                        if not subsection_content.strip():
                            continue
                        
                        # Create subsection metadata
                        subsection_metadata = DocMetadata(**asdict(section_metadata))
                        subsection_metadata.section_path.append(subsection_title)
                        
                        # Chunk the subsection
                        chunks.extend(self._chunk_section(subsection_content, subsection_metadata))
                else:
                    # No subsections, chunk the whole section
                    chunks.extend(self._chunk_section(section_content, section_metadata))
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
                for i in range(len(section_path)):
                    sections.add(" > ".join(section_path[:i+1]))
            
            # Create a hierarchical representation
            sections_text = "\n".join(sorted(list(sections)))
            
            # Find a representative doc to get basic metadata
            if docs:
                rep_doc = docs[0]
                index_content = f"MODULE: {module}\n"
                index_content += f"TITLE: {rep_doc.metadata.get('title', module)}\n"
                if rep_doc.metadata.get("deprecated"):
                    index_content += f"STATUS: {rep_doc.metadata.get('deprecated')}\n"
                
                index_content += "\nSECTIONS:\n" + sections_text
                
                # Create metadata for the index
                index_metadata = {
                    "file_path": rep_doc.metadata.get("file_path"),
                    "module_name": module,
                    "title": rep_doc.metadata.get("title", module),
                    "section_path": [],
                    "deprecated": rep_doc.metadata.get("deprecated"),
                    "is_code_example": False,
                    "is_table": False,
                    "related_modules": [],  # Could add related modules here
                    "python_version": rep_doc.metadata.get("python_version", "3.13"),
                    "content_type": "module_index",
                    "chunk_type": "module_index"
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
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        if persist_directory:
            vectorstore.persist()
        
        return vectorstore


def main():
    """Main function to demonstrate usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Python documentation for RAG")
    parser.add_argument("docs_dir", help="Path to the Python documentation directory")
    parser.add_argument("--output", "-o", help="Output directory for vector store", default="./python_docs_vectorstore")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum chunk size")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap")
    
    args = parser.parse_args()
    
    # Initialize the chunker
    chunker = PythonDocChunker(
        docs_dir=args.docs_dir,
        chunk_max_size=args.chunk_size,
        chunk_overlap=args.overlap
    )
    
    # Process all files
    print("Processing documentation files...")
    documents = chunker.process_all_files()
    print(f"Created {len(documents)} document chunks")
    
    # Build vector store
    print(f"Building vector store at {args.output}...")
    vectorstore = chunker.build_vectorstore(documents, args.output)
    print("Done!")
    
    return vectorstore


if __name__ == "__main__":
    main()
