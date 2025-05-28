# Hybrid RAG for Python Documentation

A documentation assistant that combines vector search with keyword search to answer Python questions more accurately.

```
Query: "What is asyncio.gather?"

┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│     Vector      │    │    Keyword       │    │                  │
│   Search (5)    │ +  │   Search (5)     │───▶│  Combined Answer │
│  (semantic)     │    │   (exact match)  │    │                  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
```

## Why Hybrid?

Standard RAG misses exact function names, handles typos poorly, and struggles with deprecated features. This combines:

- **Vector search**: Finds conceptually related content
- **Keyword search**: Finds exact syntax matches
- **Smart prompting**: Handles Python versioning, deprecation, and common typos

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your OPENROUTER_API_KEY and MODEL

# Run
python main.py
```

Choose "NEW" to create the database (takes ~10 minutes), then query away.

## Architecture

### Data Processing
- **507 Python 3.13 docs** → **14,506 chunks**
- Chunks by: sections, functions, code blocks, tables
- Preserves hierarchy: folder → document → section → subsection

### Search Strategy
```python
def hybrid_search(query):
    vector_results = chroma_db.similarity_search(query, k=5)
    keyword_results = whoosh_index.search(query, limit=5)
    return combine_and_rank(vector_results, keyword_results)
```

### Smart Prompting
The LLM gets context-aware instructions for:
- **Version awareness**: "3.13 is current, 3.14+ is future"
- **Deprecation handling**: Explains urgency and migration paths
- **Typo detection**: "asynchio" → "asyncio"
- **Grammar notation**: Recognizes BNF patterns in language reference

## Project Structure

```
├── python_docs_rag_chunker.py    # Document processing and chunking
├── openrouter_rag_chain.py       # Custom LLM wrapper for OpenRouter
├── main.py                       # Main application and UI
├── requirements.txt              # Dependencies
└── python_documentation/         # Source docs (download separately)
```

## Key Features

### Advanced Chunking
- Handles inconsistent header formatting across 507 files
- Preserves code blocks and tables as special structures
- Splits large sections by function definitions
- Maintains hierarchical metadata (folder/document/section/function)

### Hybrid Search
- **Chroma** (vector store) for semantic similarity
- **Whoosh** (keyword index) for exact matches
- Combined ranking prioritizes exact syntax matches

### Context-Aware Responses
- Detects grammar notation vs. actual functions
- Provides deprecation timelines with urgency context
- Suggests corrections for common Python term typos
- References specific modules and preserves exact syntax

## Example Queries

```
"What's new in asyncio for Python 3.13?"
"How do I use dataclasses with slots?"
"Is imp module deprecated?" 
"What does yeild do?" → Suggests "yield"
```

## Technical Details

### Dependencies
- **LangChain**: Document processing and text splitting
- **Chroma**: Vector storage with HuggingFace embeddings
- **Whoosh**: Full-text search engine
- **OpenRouter**: LLM API access

### Performance
- ~4000 char chunks with 200 char overlap
- BAAI/bge-large-en-v1.5 embeddings
- Sub-second search on 14K+ chunks

### Customization
Modify `PythonDocChunker` for different documentation formats:
- Adjust regex patterns for different header styles
- Change chunk sizes based on content type
- Add custom metadata extraction

## Future Enhancements

- **Deprecation weighting**: Boost newer alternatives in search results
- **Multi-format support**: Add markdown, rst, HTML documentation
- **MCP integration**: Direct integration with coding LLMs
- **Real-time updates**: Auto-sync with Python documentation releases

## Requirements

- Python 3.8+
- OpenRouter API key
- ~2GB disk space for indexes
- 8GB+ RAM recommended for processing
- Access to Python documentation in .txt - available for download on python.org

## Contributing

The chunking logic in `PythonDocChunker` is the core innovation. Extensions for other documentation formats or search improvements are welcome.

## License

MIT