# Modelos de llm
import os
from dotenv import load_dotenv 
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI # acrescentar em dependências
from langchain_huggingface import HuggingFaceEmbeddings # embeddings - na versão final armazenar os modelos no servidor
from langchain_qdrant import QdrantVectorStore, RetrievalMode # Qdrant
from langchain_qdrant.fastembed_sparse import FastEmbedSparse # Qdrant
from fastembed import (
  SparseTextEmbedding,
  TextEmbedding,
)
#from langchain_groq import ChatGroq # llm 
import concurrent.futures

load_dotenv() #lembrar para poder ler o .env

MODEL_CACHE_DIR = "./model_cache"
 
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE_DIR
os.environ["FASTEMBED_CACHE_PATH"] = os.path.join(MODEL_CACHE_DIR, "fastembed")
 
print("Cache configurado para:", MODEL_CACHE_DIR)

def mostrar_mensagem_notebook():
    print("""
Oops! Estamos com problemas por aqui. Por favor, você pode tentar novamente mais tarde?

Enquanto isso, você pode usar o nosso NotebookLM:
https://notebooklm.google.com/notebook/93d397f0-204b--4d55-93ee-8a609a6a1c79?authuser=3

Lá você pode explorar não só o conteúdo dos cadernos desenvolvidos pelo IPT e SDE,
mas também gerar mapas mentais, resumos em áudio e vídeo.
""")


def carregar_llm():

    # 1) TENTAR OLLAMA PRIMEIRO
    try:
        print("Tentando iniciar Ollama...")

        llm = ChatOllama(
            base_url="http://servidor_ollama:11434",  # ou localhost
            model="llama3.1:8b",
            temperature=0.1,
        )

        def testar_ollama():
            return llm.invoke("Explique em uma frase o que é inteligência artificial.")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(testar_ollama)
            future.result(timeout=15)  # 15 segundos

        print("Ollama carregado com sucesso.")
        return llm

    except Exception as e_ollama:
        print("Ollama falhou:", str(e_ollama))


    # # 2) TENTAR GROQ COMO SEGUNDA OPÇÃO
    try:
        print("Tentando fallback para Groq...")

        llm = ChatGroq(
            model="llama-3.1-8b-instant", # Modelo de geração mais leve, llama-3.1-8b-instant
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY"),
)

        print("Groq carregado com sucesso.")
        return llm

    except Exception as e_groq:
        print("Groq também falhou:", str(e_groq))
        mostrar_mensagem_notebook()
        raise RuntimeError("Nenhum modelo disponível.")
  
  # 3) TENTAR OPENAI COMO terceira OPÇÃO
    # try:
    #     print("Tentando fallback para OpenAI...")

    #     llm = ChatOpenAI(
    #         model="gpt-5-nano",
    #         temperature=0.1,
    #         openai_api_key=openai_key,
    #     )

    #     print("OpenAI carregado com sucesso.")
    #     return llm

    # except Exception as e_openai:
    #     print("OpenAI também falhou:", str(e_openai))
    #     mostrar_mensagem_notebook()
    #     raise RuntimeError("Nenhum modelo disponível.")
        
# Modelo denso
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("Modelo denso configurado com sucesso")
# Modelo esparso
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") #função matemática BM25, que classifica documentos com base na relevância em relação a uma consulta de pesquisa.
print("Modelo esparso configurado com sucesso")

# # Teste - deixe comentado
llm = carregar_llm()
messages = [
    (
        "system",
        "Você é um assistente muito útil e responde perguntas em uma linha.",
    ),
    ("human", "O que é Inteligência artificial?"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)

