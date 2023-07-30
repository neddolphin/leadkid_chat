from django.shortcuts import render
from django.http import JsonResponse
from .oai_queries import get_completion

import os
from django.conf import settings

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain


os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


def query_view(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')

        loader = DirectoryLoader("data/")
        documents = loader.load()
        # split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        # select which embeddings we want to use
        embeddings = OpenAIEmbeddings()
        # create the vectorestore to use as the index
        db = Chroma.from_documents(texts, embeddings)
        # expose this index in a retriever interface
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
        # create a chain to answer questions 
        qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)
        chat_history = getHistory( request )
        result = qa({"question": prompt, "chat_history": chat_history})
        answer = result['answer']

        appendToHistory( request, prompt, answer )
        return JsonResponse({'response': answer })
    return render(request, 'query.html')


def appendToHistory( request, prompt, answer ):
    request.session["history"].append( ( prompt, answer ) )
    request.session["history"] = request.session["history"][-10:]    
    # print( request.session["history"] )
    request.session.modified = True

    
def getHistory( request ):
    request.session["history"] = []
    if "history" not in request.session:
        request.session["history"] = []
        request.session.modified = True
    return request.session["history"]