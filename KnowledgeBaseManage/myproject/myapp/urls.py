from django.urls import path
from . import views

urlpatterns = [
    path('', views.knowledge_list, name='knowledge_list'),
    path('create/', views.knowledge_create, name='knowledge_create'),
    path('delete/<int:knowledge_id>/', views.knowledge_delete, name='knowledge_delete'),
]
