# start for project
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from python_docs_rag_chunker import *

print("Wellcome to the Python documetation RAG system")
print("This system is designed to help you find information in the Python documentation.")
print("\nWould you like to create a new vectorstore or load an existing one? (new/load)")
get_vector_database = insert("Please enter your choice: ")

if get_vector_database == "new":

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    # Call the function with parameters
    vectorstore = chunk_and_create_vector_database(
        docs_dir="/path/to/python-3.13-docs-text",
        output_dir="./my_vectorstore",
        chunk_size=800,
        chunk_overlap=150,
        embedding_model=embeddings
    )

    # Now you can use the vectorstore
    # For example, create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Or do a direct search
    results = vectorstore.similarity_search(
        "How to use logging in Python?", k=3)
    for doc in results:
        print(doc.page_content[:100], "...\n")


else:

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4-turbo")

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

    # Query the system
    query = "How do I set up logging to a file in Python?"
    result = qa_chain({"question": query, "chat_history": []})
    print(result["answer"])
