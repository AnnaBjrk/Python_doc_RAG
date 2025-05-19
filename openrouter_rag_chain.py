import os
import json
import requests
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional


class OpenRouterLLM(LLM):
    """Custom LLM for OpenRouter API."""
    
    model: str = "anthropic/claude-3-haiku"
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the OpenRouter API."""
        
        # Use environment variables if not provided
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        api_url = self.api_url or os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
        model = kwargs.get("model", self.model)
        
        if not api_key:
            raise ValueError("OpenRouter API key not provided")
        
        # Set up the request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False
        }
        
        # If stop sequences are provided, add them
        if stop:
            data["stop"] = stop
        
        # Make the request
        response = requests.post(
            url=api_url,
            headers=headers,
            data=json.dumps(data)
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"Error {response.status_code}: {response.text}")
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


def create_qa_chain(vectorstore):
    """Create a QA chain using OpenRouter."""
    
    load_dotenv()  # Load environment variables from .env file
    
    # Initialize the custom LLM
    llm = OpenRouterLLM(
        model=os.getenv("MODEL", "anthropic/claude-3-haiku"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_url=os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"),
        max_tokens=1000,
        temperature=0.7
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


def query_documentation(vectorstore):
    """Interactive documentation querying system."""
    
    qa_chain = create_qa_chain(vectorstore)
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
                    print(f"- File: {doc.metadata.get('file_path', 'Unknown')}")
                    print(f"- Section: {' > '.join(doc.metadata.get('section_path', ['Unknown']))}")
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


# Example usage
if __name__ == "__main__":
    from python_docs_rag_chunker import chunk_and_create_vector_database
    from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    # Check if vectorstore exists, if not create it
    import os
    if not os.path.exists("./python_docs_vectorstore"):
        print("Creating vector database...")
        vectorstore = chunk_and_create_vector_database(
            docs_dir="./python-3.13-docs-text",
            output_dir="./python_docs_vectorstore",
            chunk_size=800, 
            chunk_overlap=150,
            embedding_model=embeddings
        )
    else:
        # Load existing vectorstore
        from langchain_community.vectorstores import Chroma
        print("Loading existing vector database...")
        vectorstore = Chroma(
            persist_directory="./python_docs_vectorstore",
            embedding_function=embeddings
        )
    
    # Query the documentation
    chat_history = query_documentation(vectorstore)
