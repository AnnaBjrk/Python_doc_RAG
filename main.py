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
    max_tokens: int = 1000
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
            # Call the function with parameters
            vectorstore = current_query.chunk_and_create_vector_database()

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
                # Load existing vectorstore

                print("Loading existing vector database...")
                vectorstore = Chroma(
                    persist_directory="./my_vectorstore",
                    embedding_function=embeddings
                )

            # Query the documentation
            chat_history = query_documentation(
                vectorstore, MODEL, OPENROUTER_API_KEY, OPENROUTER_URL, max_tokens, temperature)

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
