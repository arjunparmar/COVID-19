from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from .models import Image
import cv2
from . import forms
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model,model_from_json
from keras.preprocessing.image import img_to_array
from django.contrib import messages
import base64
import io
import PIL
import numpy as np
import requests
from flask import jsonify
import os
import datetime
import time
import json
import base64
from urllib.parse import unquote
import re
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from WEB_APP.settings import BASE_DIR
from WEB_APP.settings import MEDIA_DIR
from django.core.files.base import ContentFile
from django.core.files import File
from django.http.response import StreamingHttpResponse
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import threading
import gzip
from threading import Thread, Lock
_db_lock = Lock()

def home_view(request):
    return render(request, 'home.html')

def predict_menu(request):
    return render(request, 'predict.html')

def prepare(ima):
    IMG_SIZE = 500
    # img_array = ima*255
    print('inside prepare')
    print('prepare parameter type ' + str(type(ima)))
    img_array = ima
    # img_array = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
    img_array=img_array/255.0  # filepathread in the image, convert to grayscale
    print('img_array shape:' + str(img_array.shape))    #(200,200,3)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    print('new_array shape:' + str(new_array.shape))      #(100,100,3)
    new_array =  new_array.reshape(-1,IMG_SIZE, IMG_SIZE,1)
    print('out of prepare')
    print('prepared image shape' + str(new_array.shape))
    return new_array

def predict_image(image,name_image):
    try:
        print('Inside predict_image shape :' + str(image.shape))
        model_path = os.path.join(BASE_DIR, 'covid19 densenet02.h5')
        model = load_model(model_path, compile = False)
        image1 = image.copy()
        print('name of image to be predicted is : ' + str(name_image))
        prediction = model.predict(prepare(image1))
        prediction = prediction[0]
        prediction = prediction[0]
        if prediction>=0.5 :
            prediction=1
        else:
            prediction=0
        x1=str(prediction)
        print('x1 is : ' + x1)
        _,buffer_image1 = cv2.imencode('.jpeg', image1)
        f_image1 = buffer_image1.tobytes()
        f1 = ContentFile(f_image1)
        image_file = File(f1, name = name_image )
        return image_file,x1
    except Exception as e:
        print(e)


def formpage(request):
    global flag
    global ans 
    upload_image = Image()
    modified_image = Image()
    temp_form = forms.TempForm({'predictIt':'no'})
    image_form = forms.ImageForm()
    if request.method == 'POST' :
        temp_form = forms.TempForm(request.POST)
        t_value = request.POST.get('predictIt')
        if t_value == 'yes' :
            img_obj = Image.objects.filter().order_by('-id')[0]
            name_image = img_obj.uploads.name
            test_image = img_obj.uploads
            image_bytes = test_image.read()
            target_image = PIL.Image.open(io.BytesIO(image_bytes))
            target_image = target_image.resize((500,500),PIL.Image.ANTIALIAS)
            print(type(target_image))
            image_array = np.array(target_image)
            image_file, x1 = predict_image(image_array,name_image)
            if x1 == '0':
                ans = 'You are not infected with COVID-19! You are safe!!'
            else:
                ans = 'Unfortunately you are infected with COVID-19!'
            print('Imgage_file type: ' + str(type(image_file)))
            modified_image.uploads = img_obj.uploads
            print("next step")
            modified_image.save()
            context_dict = {'form' : image_form,'temp_form' : temp_form, 'prediction' : ans, 'image_show' : modified_image}   #predicted image
        else :
            image_form = forms.ImageForm(request.POST,request.FILES)
            if image_form.is_valid():
                print('inside form.vaid')
                if request.FILES.get("uploads",None) is not None:
                    print('image prese')
                    test_image = request.FILES['uploads']
                    image_byte = test_image.read()
                    target_image = PIL.Image.open(io.BytesIO(image_byte))
                    target_image = target_image.resize((200,200),PIL.Image.ANTIALIAS)
                #   target_image = prepare(target_image, category)
                    target_image = np.array(target_image)
                    name_image = image_form.cleaned_data['uploads'].name
                #   print(type(target_image))
                #   print(target_image.shape)
                #   image_file ,x1 = predict_image(target_image,category,name_image)
                #   modified_image.uploads = image_file
                #   modified_image.save()
                    flag = 1
                    if 'uploads' in request.FILES:
                        print('inside function')
                        upload_image.uploads = request.FILES['uploads']
                        upload_image.save()
                        print('Saved image' + str(upload_image.uploads.name))
                        upload_obj = Image.objects.filter().order_by('-id')[0]
                        image_id = upload_obj.id
                        print("image id = {}".format(image_id))
                        context_dict = {'form' : image_form, 'temp_form' : temp_form, 'image_show':upload_image}     #uploaded image
                else:
                    context_dict = {'form' : image_form , 'temp_form' : temp_form}
            #return HttpResponse('The predicted class is {}'.format(x1))
            #messages.success(request,'The predicted class is {}'.format(x1))
            else:
                print(image_form.errors)

    else:
        image_form = forms.ImageForm()
        context_dict = {'form' : image_form, 'temp_form' : temp_form}
    print(context_dict)
    return render(request,'predict.html',context = context_dict)

# This API is used to predict from photo  uploaded from android device. A base64 photo is sent from device and seerver responds with prediction
@csrf_exempt
def android_predict(request):
    try : 
        print('Request method is : ' + request.method)
        if request.method == 'POST' :
            print('method is POST')
            # print('Body is : ' + str(request.body))
            decoded =  unquote(str(request.body))                   #decode URI component because Android sends using URI Encode
            # print(decoded)
            name_image = decoded.split('&')[1].split('=')[1]
            # print('name is : ' + name_image)
            b64_image = decoded.split('&')[0].split('=')[1]
            # print('Base64 image is : ' + b64_image)
            missing_padding = len(b64_image)%4
            print('Length is : ' + str(len(b64_image)))
            if missing_padding : 
                b64_image += '='*(4-missing_padding)
            # print('Modified Base64 image is : ' + b64_image)
            # print('New length is : ' + str(len(b64_image)))
            image = PIL.Image.open(io.BytesIO(base64.b64decode(b64_image)))
            target_image = image.resize((500,500),PIL.Image.ANTIALIAS)
            # print(type(target_image))
            image_array = np.array(target_image)
            image_file, x1 = predict_image(image_array,name_image)
            # image = PIL.Image.open(io.BytesIO(b64_image))
            context_dict = {'statusCode' : 0, 'statusMessage' : 'working', 'prediction' : x1}
        else :
            print('method is GET')
            context_dict = {'statusCode' : 1, 'statusMessage' : 'request type is not post'}
        print(context_dict)
        return JsonResponse(context_dict, safe=False)
    except Exception as e: 
        print(e)
        context_dict = {'statusCode' : 1, 'statusMessage' : 'exception occurred'}
        print(context_dict)
        return JsonResponse(context_dict, safe=False)
