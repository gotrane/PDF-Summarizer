from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community import faiss
from langchain_community.vectorstores.faiss import FAISS

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# from vector import retriever
from pypdf import PdfReader

template = """
Voici des extraits d'un document PDF. Résume-les en 3 à 5 phrases claires et complètes.

EXTRAITS:
{context}

RÉSUMÉ:
"""

def process_text (text) : 
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    #turns each text chunk into a 384-dimensional embedding that captures its semantic meaning
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    knowledgeBase = FAISS.from_texts(chunks,embeddings)

    return knowledgeBase

def summarizer (pdf) : 
    if pdf is not None : 
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        knowledgeBase = process_text(text)

        query = "Summarize the content of the PDF file in approximately 3-5 sentences."

        if query : 
            docs = knowledgeBase.similarity_search(query,k=5) #key information the embedding where the vector are similar to the vector of the query
            context = "\n\n".join(doc.page_content for doc in docs)
            # OpenAIModel = "gpt-3.5-turbo-16k"
            prompt = ChatPromptTemplate.from_template(template)
            model = OllamaLLM(model="llama3.2")
            chain = prompt | model
            # llm = ChatOpenAI(model=OpenAIModel,temperature=0)
            # reviews = retriever.invoke(query)
            result = chain.invoke({"context": context})
            #chain = load_qa_chain(llm,chain_type='stuff')

            # response = chain.run(input_documents=docs,question=query)

            return result