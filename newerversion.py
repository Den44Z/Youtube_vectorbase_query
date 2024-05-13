import sys
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()
key = ''

def create_db_from_youtube_video_url(video_url: str, embeddings) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, embeddings, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-4-turbo",
                     openai_api_key=key)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_script.py <youtube_video_url> <question>")
        sys.exit(1)

    youtube_url = sys.argv[1]
    question = sys.argv[2]

    embeddings = OpenAIEmbeddings(openai_api_key=key)
    db = create_db_from_youtube_video_url(youtube_url, embeddings)
    response, _ = get_response_from_query(db, question, embeddings)
    print(response)
