from django.urls import path
from .views import ScannerView


urlpatterns = [
    path('scan/', ScannerView.as_view(), name='scanner'),

]