from django.shortcuts import render
from django.http import JsonResponse
from .oai_queries import get_completion

def query_view(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')
        prompt_with_history = getPromptWithHistory( request, prompt )
        # print( prompt_with_history )
        response = get_completion( prompt_with_history )
        print( response )
        appendToHistory( request, prompt, response )
        return JsonResponse({'response': response})
    return render(request, 'query.html')

def appendToHistory( request, prompt, response ):
    request.session["history"].extend( [ { "role": "user", "content": prompt }, 
                                         { "role": "assistant", "content": response }, ] )
    print( request.session["history"] )
    request.session.modified = True

def getPromptWithHistory( request, prompt ):
    if "history" in request.session:
        return request.session["history"] + [{ "role": "user", "content": prompt }]
    else:
        init_string = [ {"role": "system", "content": "You are a helpful assistant."}, ]
        request.session["history"] = init_string
        return init_string + [{ "role": "user", "content": prompt }]