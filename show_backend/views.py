from django.http import HttpResponse
from django.shortcuts import render
from pathlib import Path

def index(request):
    # return HttpResponse("APIs for ShowOnMe are running!")
    # print(Path(__file__))
    return render(request, 'index.html')