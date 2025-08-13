from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_document, name='upload_document'),
    path('download-pdf/', views.download_pdf, name='download_pdf'),
]
