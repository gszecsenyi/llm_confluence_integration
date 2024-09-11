import os
from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class ConfluenceQA:
    def __init__(self,config:dict = {}):
        self.config = config
        print("config",self.config)
        self.qa = None
        self.openai_api_key = config.get("openai_api_key",None)
        print("X",self.openai_api_key)
        self.embedding = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.llm = ChatOpenAI(model_name=self.config.get("model_name",None), temperature=0.0, openai_api_key=self.openai_api_key)
        self.vectordb = None
        self.retriever = None
        self.vector_db_confluence_docs(False)
        
    
    def vector_db_confluence_docs(self,force_reload:bool= False) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        persist_directory = self.config.get("persist_directory",None)
        confluence_url = self.config.get("confluence_url",None)
        username = self.config.get("username",None)
        print("username",username)
        api_key = self.config.get("api_key",None)
        print("api_key",api_key)
        space_key = self.config.get("space_key",None)
        if persist_directory and os.path.exists(persist_directory) and not force_reload:
            ## Load from the persist db
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        else:
            ## 1. Extract the documents
            loader = ConfluenceLoader(
                url=confluence_url,
                # username = username,
                cloud=False,
                token= api_key,
                space_key="PD",
                limit=50,
            )
            documents = loader.load()
            ## 2. Split the texts
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
            texts = text_splitter.split_documents(texts)

            ## 3. Create Embeddings and add to chroma store
            ##TODO: Validate if self.embedding is not None
            self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=persist_directory)
        
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k":4})
        
        
    def answer_confluence(self,question:str) ->str:
        """
        Answer the question
        """
        context = "You are a confluence user and you want to know the answer to the question"
        self.system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Context: {context}"
        )
        input = question
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.chain = create_retrieval_chain(self.retriever, question_answer_chain)
        return self.chain.invoke({"input": question}).get("answer","")