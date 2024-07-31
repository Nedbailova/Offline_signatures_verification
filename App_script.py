#!/usr/bin/env python
# coding: utf-8

# In[19]:

import warnings
import cv2 as cv
import numpy as np
from math import atan2
import PIL.Image
import os
from PIL import Image
import os
import sys

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

warnings.filterwarnings("ignore")


# In[20]:


#чтение изображения
def read_img(path):
    return cv.imread(str(path))


# In[21]:


def bradley_binarization(image, threshold=15):
    window_size=image.shape[1] // 4
    
    # Преобразование изображения в оттенки серого
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Вычисление интегрального изображения
    integral_image = cv.integral(gray)

    
    # Определение смещения для размера окна
    S = window_size // 2
    
    
    # Создание выходного изображения, инициализированного нулями
    output = np.zeros_like(gray)

    
    # Перебор каждого пикселя на изображении
    for y in range(gray.shape[0]): #H
        for x in range(gray.shape[1]): #W
            # Определение области интереса (ROI) для текущего пикселя
            y1 = max(0, y - S)
            y2 = min(gray.shape[0] - 1, y + S)
            x1 = max(0, x - S)
            x2 = min(gray.shape[1] - 1, x + S)
            
            # Вычисление площади текущего окна
            count = (y2 - y1) * (x2 - x1)
            
            # Вычисление суммы значений пикселей внутри окна с использованием интегрального изображения
            window_sum = integral_image[y2, x2] - integral_image[y2, x1] -                          integral_image[y1, x2] + integral_image[y1, x1]
            
            # Вычисление порогового значения для текущего окна
            threshold_value = (1 - threshold / 100.0) * window_sum / count
            
            # Бинаризация пикселя на основе порогового значения
            if gray[y, x] > threshold_value:
                output[y, x] = 255
                
    return output


# In[22]:


def rotation(image_mf):
    #Бинарное изображение
    img = image_mf

    #Координаты черных пикселей
    y, x = np.where(img == 0)

    #Среднее по координатам
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    #Централизация
    x_centered = x - mean_x
    y_centered = y - mean_y


    #Матрица ковариаций
    covariance_matrix = np.cov(np.array([x_centered, y_centered]))

    #Собственные числа и вектора матрицы ковариаций
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)


    #Сортировка собственных векторов
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigen_vecs = eigenvectors[:, sort_indices]

    #строки м-цы А состоят из собственных векторов м-цы ковариации в порядке убывания собственных чисел
    A = np.zeros((2,2))

    A[0][0] = eigenvectors[0][0]
    A[0][1] = eigenvectors[1][0]
    A[1][0] = eigenvectors[0][1]
    A[1][1] = eigenvectors[1][1]

#     print("angle 1 = ", abs(atan2(A[0][1],A[0][0])*180/np.pi))
#     print("angle 2 = ", abs(atan2(A[1][1],A[1][0])*180/np.pi))

    if(abs(atan2(A[0][1],A[0][0])*180/np.pi)>90):
        A[0][0] = eigenvectors[0][1]
        A[0][1] = eigenvectors[1][1]
        A[1][0] = eigenvectors[0][0]
        A[1][1] = eigenvectors[1][0]

        #Y = A(x-m)
        y1 = np.dot(A, np.array([x_centered, y_centered]))
       
        y1 = y1.astype(int)
        #сдвиг изображения
        if(y1[0].min()<0):
            y1[0] -= y1[0].min()
        if(y1[1].min()<0):
            y1[1] -= y1[1].min()

        #создаем шаблон изображения по максимальным значениям координат
        a1 = np.zeros((y1[1].max(),y1[0].max()))
        a1[...] = 255
        for i in range(y1.shape[1]):
            if (0<y1[0][i]<y1[0].max()) and (0<y1[1][i]<y1[1].max()):
                a1[y1[1][i]][y1[0][i]] = 0
    else:

        #Y = A(x-m)
        y1 = np.dot(A, np.array([x_centered, y_centered]))
        
        #перевод в интовые координаты
        y1 = y1.astype(int)
        
        #сдвиг изображения, чтобы убрать отрицательные значения координат
        if(y1[0].min()<0):
            y1[0] -= y1[0].min()
        if(y1[1].min()<0):
            y1[1] -= y1[1].min()
       
        #создаем шаблон изображения по максимальным значениям координат
#         a1 = np.zeros((y1[1].max()+1,y1[0].max()+1))
        a1 = np.zeros((y1[1].max(),y1[0].max()))
        a1[...] = 255
        for i in range(y1.shape[1]):
            if (0<y1[0][i]<y1[0].max()) and (0<y1[1][i]<y1[1].max()):
                a1[y1[1][i]][y1[0][i]] = 0



    H, L = np.shape(a1) #ширины и высота изображения
    return a1, H, L


# In[23]:


#изменение размера
def resize_image(img, target_size=(224, 224)):
    # Открытие изображения
    fill_color = (255, 255, 255)
    img = Image.fromarray(img)
    
    
    # Получение размеров исходного изображения
    original_size = img.size
    
    # Вычисление коэффициентов масштабирования
    width_ratio = target_size[0] / original_size[0]
    height_ratio = target_size[1] / original_size[1]
    
    # Выбор минимального коэффициента масштабирования, чтобы сохранить пропорции изображения
    min_ratio = min(width_ratio, height_ratio)
    
    # Новые размеры изображения
    new_width = int(original_size[0] * min_ratio)
    new_height = int(original_size[1] * min_ratio)
    
    # Масштабирование изображения с сохранением пропорций
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Создание белого холста нужного размера
    background = Image.new('RGB', target_size, fill_color)
    
    # Размещение масштабированного изображения по центру белого холста
    offset = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    background.paste(img, offset)
    
    return background


def resize_image1(img, target_size=(512, 512)):
    # Открытие изображения
    fill_color = (255, 255, 255)
    img = Image.fromarray(img)
    
    
    # Получение размеров исходного изображения
    original_size = img.size
    
    # Вычисление коэффициентов масштабирования
    width_ratio = target_size[0] / original_size[0]
    height_ratio = target_size[1] / original_size[1]
    
    # Выбор минимального коэффициента масштабирования, чтобы сохранить пропорции изображения
    min_ratio = min(width_ratio, height_ratio)
    
    # Новые размеры изображения
    new_width = int(original_size[0] * min_ratio)
    new_height = int(original_size[1] * min_ratio)
    
    # Масштабирование изображения с сохранением пропорций
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Создание белого холста нужного размера
    background = Image.new('RGB', target_size, fill_color)
    
    # Размещение масштабированного изображения по центру белого холста
    offset = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    background.paste(img, offset)
    
    return background

# In[26]:


# Получение аргумента командной строки - путь файла
file_path = sys.argv[1]
# file_path="../signatures/CEDAR/CEDAR/19/original_19_4.png"


# In[25]:


resnet50 = tv.models.resnet50(weights=None)
num_classes = 55
# resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
resnet50.fc = nn.Sequential(
    nn.Linear(resnet50.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # Пример добавления dropout
    nn.Linear(512, num_classes)
)

base_path = os.path.dirname(os.path.abspath(__file__))  # Путь к директории, где находится исполняемый файл
weights_path = os.path.join(base_path,'resnet50_epoch_80.pt')
resnet50.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

def load_image(image_path):
    image_filepath = str(image_path)
    #предобработка изображения
    img_r = read_img(str(image_filepath)) #чтение
    img_b = bradley_binarization(img_r) #бинаризация
    img_mf = cv.medianBlur(img_b, 3) #медианный фильтр
    img_r, a, b = rotation(img_mf)
    img = resize_image(img_r) #масштабирование
    img = np.float32(img)
    img = img/255.0
    img = img.transpose((2, 0, 1))
    t_img = torch.from_numpy(img)
    t_img = t_img.unsqueeze(0)
    return t_img



image_filepath = str(file_path)
    #предобработка изображения
img_r1 = read_img(str(image_filepath)) #чтение
img_b1 = bradley_binarization(img_r1) #бинаризация
img_mf1 = cv.medianBlur(img_b1, 3) #медианный фильтр
img_r1, a, b = rotation(img_mf1)
img1 = resize_image1(img_r1) #масштабирование
output_filename = os.path.basename(image_filepath).replace('.png', '_processed.png')
output_path = os.path.join(os.path.dirname(image_filepath), output_filename)
img1.save(output_path)

resnet50.eval()  

image = load_image(file_path)  



with torch.no_grad():
    output = resnet50(image)


predicted_class = torch.argmax(output, dim=1).item()+1


# In[ ]:

with open("result.txt", "w") as file:
    file.write(str(predicted_class))




