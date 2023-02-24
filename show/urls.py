from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('get_images/', views.get_images, name='get_images'),
]