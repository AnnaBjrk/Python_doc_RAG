# start for project
# /Users/annasmac/Documents/Nackademin/MachineLearning projects/
# ML_project_RAG/main.py:120: LangChainDeprecationWarning:
# The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0.
# An updated version of the class exists in the :class:`~langchain-huggingface package and should be used instead.
# To use it run `pip install -U :class:`~langchain-huggingface` and import as `from :class:`~langchain_huggingface
# import HuggingFaceEmbeddings``.


# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from python_docs_rag_chunker import PythonDocChunker, DocMetadata
from openrouter_rag_chain import OpenRouterLLM
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from datetime import datetime

# If WhooshSearch is in python_docs_rag_chunker.py:
from python_docs_rag_chunker import WhooshSearch


def hybrid_search_and_query(vectorstore, whoosh_search, question, llm, chat_history):
    """
    Perform hybrid search using both vector and keyword search, then query LLM.
    """
    # Get vector search results
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    vector_docs = vector_retriever.get_relevant_documents(question)

    # Get keyword search results
    whoosh_results = whoosh_search.search(question, limit=5)

    # Format results
    vector_context = "\n".join([
        f"Source: {doc.metadata.get('module_name', 'Unknown')} - {doc.metadata.get('section_path', '')}\n"
        f"Content: {doc.page_content[:300]}...\n"
        for doc in vector_docs
    ])

    keyword_context = "\n".join([
        f"Source: {result['document']} - {result['section_path']}\n"
        f"Content: {result['content'][:300]}...\n"
        for result in whoosh_results
    ])

    # Build chat history context
    history_context = ""
    if chat_history:
        recent_history = chat_history[-2:]  # Last 2 exchanges
        history_context = "Previous conversation:\n" + "\n".join([
            f"Q: {q}\nA: {a[:100]}...\n" for q, a in recent_history
        ])

    # Create comprehensive prompt
    prompt = f"""You are a Python documentation assistant. Use the following search results to answer the question.

{history_context}

SEMANTIC SEARCH RESULTS (related concepts):
{vector_context}

KEYWORD SEARCH RESULTS (exact matches):
{keyword_context}

Question: {question}

IMPORTANT GUIDELINES:

**Grammar Notation Recognition:**
If the user asks about terms with underscores (like set_display, list_display, function_def), 
these might be grammar notation from Python's language reference, not functions.
Look for BNF notation patterns like "term ::= definition" in the search results.

**Python Version Context (Current: 3.13):**
When interpreting version information:
- Versions 3.12 and below: Already released, past versions
- Version 3.13: Current version (what users are assumed to be using)
- Version 3.14 and above: Future versions, not yet released

For deprecation timelines, always contextualize:
- "Deprecated since X.Y" - explain how long it's been deprecated
- "Planned removal in X.Y" - explain how urgent the migration is

**Spelling Error Detection:**
If your confidence in this answer is below 70% (especially due to not finding relevant information), 
check if the question contains potential spelling errors for technical terms like:
- Module names (e.g., "asynchio" → "asyncio")
- Function names (e.g., "lenght" → "length") 
- Method names (e.g., "apend" → "append")
- Keywords (e.g., "yeild" → "yield")

If you suspect spelling errors, suggest the correct spelling and search again mentally.

**Response Instructions:**
- Use information from both search results - prioritize exact matches for syntax
- If keyword results show exact syntax, include it
- If semantic results provide better explanations, use those
- Mention specific modules/functions when relevant
- For code examples, preserve exact syntax from the documentation
- Keep answers concise but complete
- Always provide practical examples when possible

Answer:"""

    # Query the LLM directly
    response = llm._call(prompt)
    return response, vector_docs + [{"source": "keyword", "content": r} for r in whoosh_results]


def query_documentation_hybrid(vectorstore, whoosh_search, MODEL, OPENROUTER_API_KEY, OPENROUTER_URL, max_tokens, temperature):
    """Modified query function using hybrid search."""

    # Initialize LLM
    llm = OpenRouterLLM(
        model=MODEL,
        api_key=OPENROUTER_API_KEY,
        api_url=OPENROUTER_URL,
        max_tokens=max_tokens,
        temperature=temperature
    )

    chat_history = []

    print("\nWelcome to the Python Documentation Assistant (Hybrid Search)!")
    print("Ask any question about Python or type 'exit' to quit.\n")

    while True:
        query = input("\nYour question: ")

        if query.lower() in ["exit", "quit", "q"]:
            print("\nThank you for using the Python Documentation Assistant!")
            break

        try:
            # Use hybrid search
            answer, sources = hybrid_search_and_query(
                vectorstore, whoosh_search, query, llm, chat_history
            )

            print("\nAnswer:")
            print(answer)

            # Show sources if requested
            if "sources" in query.lower() or "references" in query.lower():
                print("\nSources used:")
                for i, source in enumerate(sources[:5]):  # Show first 5
                    if isinstance(source, dict) and source.get("source") == "keyword":
                        print(f"\nKeyword Source {i+1}:")
                        print(f"- Document: {source['content']['document']}")
                        print(
                            f"- Section: {source['content']['section_path']}")
                    else:
                        print(f"\nVector Source {i+1}:")
                        print(
                            f"- File: {source.metadata.get('file_path', 'Unknown')}")
                        print(
                            f"- Section: {source.metadata.get('section_path', '')}")

            # Update chat history
            chat_history.append((query, answer))

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")

    return chat_history


def create_qa_chain(vectorstore, MODEL, OPENROUTER_API_KEY, OPENROUTER_URL, max_tokens, temperature):
    """Create a QA chain using OpenRouter."""

    # Initialize the custom LLM
    llm = OpenRouterLLM(
        model=MODEL,
        api_key=OPENROUTER_API_KEY,
        api_url=OPENROUTER_URL,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # Create a retriever from the vectorstore
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Create the conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain


def query_documentation(vectorstore, MODEL, OPENROUTER_API_KEY, OPENROUTER_URL, max_tokens, temperature):
    """Interactive documentation querying system."""

    qa_chain = create_qa_chain(
        vectorstore, MODEL, OPENROUTER_API_KEY, OPENROUTER_URL, max_tokens, temperature)
    chat_history = []

    print("\nWelcome to the Python Documentation Assistant!")
    print("Ask any question about Python or type 'exit' to quit.\n")

    while True:
        # Get user query
        query = input("\nYour question: ")

        # Check if user wants to exit
        if query.lower() in ["exit", "quit", "q"]:
            print("\nThank you for using the Python Documentation Assistant!")
            break

        try:
            # Query the system
            result = qa_chain({
                "question": query,
                "chat_history": chat_history
            })

            # Print the answer
            print("\nAnswer:")
            print(result["answer"])

            # Show source documents if requested
            if "sources" in query.lower() or "references" in query.lower():
                print("\nSources:")
                for i, doc in enumerate(result["source_documents"]):
                    print(f"\nSource {i+1}:")
                    print(
                        f"- File: {doc.metadata.get('file_path', 'Unknown')}")
                    # Update this line to handle section_path as string
                    section_path = doc.metadata.get('section_path', '')
                    if section_path:
                        print(f"- Section: {section_path}")
                    else:
                        print(f"- Section: Unknown")

                    if doc.metadata.get('is_code_example', False):
                        print(f"- Type: Code Example")
                    elif doc.metadata.get('is_table', False):
                        print(f"- Type: Table")
                    print(f"- Content: {doc.page_content[:150]}...")

            # Update chat history
            chat_history.append((query, result["answer"]))

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")

    return chat_history


def main():

    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    MODEL = os.getenv("MODEL")
    OPENROUTER_URL = os.getenv("OPENROUTER_URL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    max_tokens: int = 3000
    temperature: float = 0.7

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docs_dir = "./python_documentation/python-3.13-docs-text"
    print("\n*******************************************\n")
    print("Wellcome to the Python documetation RAG system")
    print("\n*******************************************\n")
    print("This system is designed to help you find information in the Python documentation.")

    running = True
    while running == True:
        print("\n* Would you like to create a new vectorstore or load an existing one? ")
        print("- type NEW and press enter/\n")
        print(
            "* If you want to start a query with an existing vectorstore \n- just press enter\n")
        print("* If you want to quit \n- type EXIT and press enter\n")
        choice = input("Please enter your choice: ")
        if choice.lower() == "new":

            # Initialize embedding model
            current_query = PythonDocChunker(docs_dir=docs_dir)

            # Create both vector store and Whoosh index
            vectorstore, whoosh_index = current_query.chunk_and_create_vector_database(
                create_whoosh=True)
            print("Both vectorstore and Whoosh index created")

            print("Vectorstore created and saved to ./my_vectorstore")
            # Next steps
            print(
                "You can now use the vectorstore to search for information in the Python documentation.\n")

        elif choice.lower() == "exit":
            print("Exiting the program.")
            running = False
        else:
            if not os.path.exists("./my_vectorstore"):
                print("No vectorstore database. Create one first")
                continue

            else:
                print("Loading existing vector database...")
                vectorstore = Chroma(
                    persist_directory="./my_vectorstore",
                    embedding_function=embeddings
                )

                # Check if Whoosh index exists and use hybrid search
                if os.path.exists("./my_whoosh_index"):
                    whoosh_search = WhooshSearch("./my_whoosh_index")
                    print("Loaded Whoosh keyword index")

                    # Use HYBRID search
                    chat_history = query_documentation_hybrid(
                        vectorstore, whoosh_search, MODEL, OPENROUTER_API_KEY,
                        OPENROUTER_URL, max_tokens, temperature
                    )
                else:
                    print("No Whoosh index found. Using vector search only...")
                    # Fall back to original method
                    chat_history = query_documentation(
                        vectorstore, MODEL, OPENROUTER_API_KEY, OPENROUTER_URL, max_tokens, temperature
                    )

            # Save the chat history to a file
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"chat_history_archive/chat_history_{current_datetime}.txt"
            os.makedirs("chat_history_archive", exist_ok=True)
            with open(filename, "w") as f:
                for question, answer in chat_history:
                    f.write(f"Q: {question}\n")
                    f.write(f"A: {answer}\n\n")
            print("Chat history saved to chat_history_archive folder")

    print("\nThank you for using the Python Documentation Assistant!")


# Example usage
if __name__ == "__main__":
    main()
