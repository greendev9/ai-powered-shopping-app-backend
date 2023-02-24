"""show_backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from . import views
from django.urls import path, re_path
from django.shortcuts import render
from django.utils._os import safe_join
from pathlib import Path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve, directory_index

def render_react(request):
    return render(request, "index.html")

def re_serve(request, path, document_root=None, show_indexes=True):
    fullpath = Path(safe_join(settings.STATIC_ROOT, path))
    return directory_index(path, fullpath)

urlpatterns = [
    path('show/', include('show.urls')),
    path('admin/', admin.site.urls),
    # re_path(r"^$", render_react),
    # re_path(r"^(?:.*)/?$", render_react),
]

# urlpatterns += [
#     re_path(r'^static/(?P<path>.*)$', re_serve, {
#         'document_root': settings.STATIC_ROOT,
#         'show_indexes': 'True'
#     }),
# ]
