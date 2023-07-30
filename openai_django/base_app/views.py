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
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI



os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


def query_view(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')

        if os.path.exists("persist"):
            print("Reusing index...\n")
            vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
            index = VectorStoreIndexWrapper(vectorstore=vectorstore)
        else:
            #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
            loader = DirectoryLoader("data/")
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
         
        chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model="gpt-3.5-turbo"),
                retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
                )
        
        result = chain( { "question": prompt, "chat_history": listToTuple( getHistory( request ) ) } )
        answer = result['answer']

        appendToHistory( request, prompt, answer )
        return JsonResponse({'response': answer })
    return render(request, 'query.html')


def appendToHistory( request, prompt, answer ):
    request.session["history"].append( ( prompt, answer ) )
    request.session["history"] = request.session["history"][-10:]    
    print( request.session["history"] )
    request.session.modified = True

    
def getHistory( request ):
    # request.session["history"] = []
    # request.session.modified = True
    if "history" not in request.session:
        request.session["history"] = []
        request.session.modified = True
    return request.session["history"]

def listToTuple( x ):
    y = []
    for item in x:
        y.append( tuple( item ) )
    return y