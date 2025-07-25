import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit as st
import google.generativeai as genai
from utils import extract_text, get_summary_from_gemini, MODEL_NAME, extract_text_from_url, get_text_chunks, get_vector_store
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Get the Gemini API key from Streamlit secrets
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Configure the Gemini API
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    st.warning("Please add your Gemini API key to the Streamlit secrets to use this app.")
    st.stop()

def main():
    st.set_page_config(page_title="Chat with your documents", page_icon=":books:", layout="wide")
    st.header("Chat with your documents :books:")

    # Initialize session state variables correctly
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="file_uploader")
        
        st.subheader("From the Web")
        url = st.text_input("Enter a URL", key="url_input")

        if st.button("Process"):
            all_chunks = []
            # Process URL first if provided and not already processed
            if url and url not in st.session_state.processed_files:
                with st.spinner("Fetching and processing URL..."):
                    text = extract_text_from_url(url)
                    all_chunks.extend(get_text_chunks(text, url))
                    st.session_state.processed_files.add(url)
                    st.session_state.chat_history.append(("assistant", f"I've finished reading the content from {url}."))
                    st.success("URL processed successfully!")
            
            # Process uploaded files
            if uploaded_files:
                with st.spinner("Processing uploaded files..."):
                    for file in uploaded_files:
                        if file.name not in st.session_state.processed_files:
                            text = extract_text(file)
                            all_chunks.extend(get_text_chunks(text, file.name))
                            st.session_state.processed_files.add(file.name)
                    st.session_state.chat_history.append(("assistant", "I've finished reading the uploaded documents."))
                    st.success("Files processed successfully!")

            if all_chunks:
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = get_vector_store(all_chunks, gemini_api_key)
                else:
                    st.session_state.vector_store.add_documents(all_chunks)
                
                st.session_state.chat_history.append(("assistant", "The new information has been added. How can I help you?"))
                st.rerun()

        

            if st.button("Summarize All Documents"):
                with st.spinner("Generating summary..."):
                    try:
                        full_text = "".join([doc.page_content for doc in st.session_state.vector_store.docstore._dict.values()])
                        summary = get_summary_from_gemini(full_text, gemini_api_key)
                        st.session_state.chat_history.append(("assistant", f"Of course! Here is a summary of all documents:\n\n{summary}"))
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")

    if not st.session_state.chat_history:
        st.info("Please upload one or more files or enter a URL to start chatting.")
    else:
        # Display chat history
        for author, message in st.session_state.chat_history:
            if isinstance(message, dict):
                with st.chat_message(author):
                    st.write(message["answer"])
                    if message["sources"]:
                        with st.expander("Sources"):
                            st.write(message["sources"])
            else:
                with st.chat_message(author):
                    st.write(message)

        # Chat input
        user_question = st.chat_input("Ask a question about the documents")

        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.write(user_question)

            with st.spinner("Generating answer..."):
                try:
                    # Define a clear prompt template to guide the model's behavior
                    from langchain.prompts import PromptTemplate
                    prompt_template = """
                    You are a helpful assistant for question-answering tasks.
                    Use the following pieces of retrieved context to answer the question.
                    If you don't know the answer, just say that you don't know.
                    Provide a concise answer, but include all relevant details from the context.

                    Context:
                    {summaries}

                    Question:
                    {question}

                    Answer:
                    """
                    
                    # Create a PromptTemplate object
                    PROMPT = PromptTemplate(
                        template=prompt_template, input_variables=["summaries", "question"]
                    )

                    # Configure the LLM and the retrieval chain
                    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3, google_api_key=gemini_api_key)
                    
                    # Use from_chain_type to pass the custom prompt
                    chain = RetrievalQAWithSourcesChain.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever(),
                        chain_type_kwargs={"prompt": PROMPT},
                    )
                    
                    # Call the chain with the user's question
                    response = chain({"question": user_question}, return_only_outputs=True)
                    
                    # Append the answer and sources to the history
                    st.session_state.chat_history.append(("assistant", response))
                    with st.chat_message("assistant"):
                        st.write(response["answer"])
                        if response["sources"]:
                            with st.expander("Sources"):
                                st.write(response["sources"])
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

if __name__ == '__main__':
    main()