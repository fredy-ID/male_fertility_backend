from rest_framework import serializers
from .models import Scanner

class ScannerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scanner
        fields = 'all'