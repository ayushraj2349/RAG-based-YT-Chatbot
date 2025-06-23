
import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize the RAG system
def init_rag_system(video_id, language="en"):
    try:
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, 
            languages=[language]
        )
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = splitter.create_documents([transcript])
        
        # Create vector store
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
        
        # Create LLM and prompt
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        prompt = PromptTemplate(
            template="""
            You are a helpful YouTube video assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.
            
            Context: {context}
            
            Question: {question}
            """,
            input_variables=['context', 'question']
        )
        
        # Create RAG chain
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        rag_chain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            })
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain
        
    except TranscriptsDisabled:
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def main():
    st.title("YouTube Video Chatbot")
    st.subheader("Chat with any YouTube video using AI")
    
    # Video input
    video_id = st.text_input("Enter YouTube Video ID:", placeholder="Gfr50f6ZBvo")
    language = st.text_input("Language Code:", value="en", placeholder="en")
    
    if st.button("Process Video"):
        if not video_id:
            st.warning("Please enter a YouTube Video ID")
            return
        
        with st.spinner("Processing video transcript..."):
            rag_chain = init_rag_system(video_id, language)
            
            if rag_chain is None:
                st.error("Failed to process video. Check if captions are available.")
                return
            
            st.session_state.rag_chain = rag_chain
            st.session_state.video_processed = True
            st.success("Video processed successfully! You can now chat with the video.")
    
    # Chat interface
    if st.session_state.get("video_processed", False):
        st.divider()
        st.header("Chat with the Video")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Accept user input
        if prompt := st.chat_input("Ask about the video..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get response from RAG chain
                        response = st.session_state.rag_chain.invoke(prompt)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
