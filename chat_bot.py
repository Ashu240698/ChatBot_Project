import streamlit
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "Enter your API Key"

# Created a header
streamlit.header("OpenAI powered Chat Bot")

# Created a sidebar for file uploading
with streamlit.sidebar:
    trainingDataFiles = streamlit.file_uploader("Upload the training data here.")

# Extract data from file if file uploaded
if trainingDataFiles:
    fileData = PdfReader(trainingDataFiles)
    textData = ""

    for page in fileData.pages:
        textData += page.extract_text()
        # streamlit.write(textData)

    textSplittedData = RecursiveCharacterTextSplitter(
        separators=list("\n"),
        length_function=len,
        chunk_size=1000,
        chunk_overlap=150
    )

    # Data chunks created on specified conditions
    dataChunks = textSplittedData.split_text(textData)
    # streamlit.write(dataChunks)

    # Embeddings generated
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Vector DB created to store the data chunks and embeddings
    vectorDB = FAISS.from_texts(dataChunks, embeddings)

    userQuestion = streamlit.text_input("How can I help you??")

    if userQuestion:
        # Performed a semantic search to get the best possible matching data chunk from vector DB
        matchingChunks = vectorDB.similarity_search(userQuestion)
        # streamlit.write(matchingChunks)

        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        chain = load_qa_chain(llm=llm, chain_type="stuff")
        answer = chain.run(input_documents=matchingChunks, question=userQuestion)
        streamlit.write(answer)
