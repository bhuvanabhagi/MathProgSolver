from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  
    path('graphical/', views.graphical_method, name='graphical_method'),
]
