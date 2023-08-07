import os
from rest_framework import generics, status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image  # Import PIL module
import numpy as np
from django.core.files.base import ContentFile
import requests
from .ai_models import model_1_15_epochs
from .forms import UploadFileForm
import tempfile
import traceback

from .serializers import ScannerSerializer

class_names = ['amorphous', 'normal', 'pyriform', 'tapered']
num_classes = len(class_names)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                        input_shape=(180,
                                    180,
                                    3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

try:
    model = load_model('scanner/ai_models/model_1_15_epochs')
except Exception as e:
    print('_____________________________')
    print('Erreur lors du chargement du modèle :', e)
    traceback.print_exc()  # Ajoutez cette ligne pour afficher la trace de la pile
    print('_____________________________')
    raise

class ScannerView(generics.CreateAPIView):
    serializer_class = ScannerSerializer
    permission_classes = (AllowAny,)
    model = model_1_15_epochs
    class_names = ['amorphous', 'normal', 'pyriform', 'tapered']

    def create(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')  # Use request.FILES instead of request.POST
        
        if image_file:
            image = Image.open(image_file)  # Use PIL's Image.open to open the image
            image = image.resize((180, 180))  # Resize the image
            image_array = img_to_array(image)
            image_array = image_array / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            
            prediction = model.predict(image_array)
            
            probs = np.exp(prediction) / np.sum(np.exp(prediction), axis=1, keepdims=True)
            
            predicted_class_index = np.argmax(probs, axis=1)
            predicted_class_name = self.class_names[predicted_class_index]
            
            return Response(
                {
                    'predicted_class': predicted_class_name,
                    'class_probabilities': prediction.tolist()
                }, 
                status=status.HTTP_200_OK
            )
        else:
            return Response(
                {
                    'error': "Aucune image n'a été envoyée"
                }, 
                status=status.HTTP_400_BAD_REQUEST
            )
