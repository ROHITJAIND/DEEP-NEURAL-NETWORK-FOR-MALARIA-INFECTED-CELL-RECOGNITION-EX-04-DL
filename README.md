# EX-04 Deep Neural Network for Malaria Infected Cell Recognition

### Aim:
To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.  **DATE: 11-11-2024**
### Problem Statement and Dataset:
The task is to automatically classify red blood cell images into two categories: parasitized (malaria-infected) and uninfected (healthy). Malaria-infected cells contain the Plasmodium parasite, while uninfected cells are healthy. The goal is to build a convolutional neural network (CNN) to accurately distinguish between these classes.
Manual inspection of blood smears is time-consuming and prone to errors. By using deep learning, we can automate the process, speeding up diagnosis, reducing healthcare professionals' workload, and improving detection accuracy.
The dataset consists of 27,558 annotated cell images, evenly split between parasitized and uninfected cells, providing a reliable foundation for model training and testing.
### Neural Network Model
![EX-04-DL-OUTPUT (5)](https://github.com/user-attachments/assets/5d23b1b2-a0a7-4bfb-9c03-c2a768878aca)

### Design Steps
1. **Import Libraries**:Import TensorFlow, data preprocessing tools, and visualization libraries.
2. **Configure GPU**:Set up TensorFlow for GPU acceleration to speed up training.
3. **Data Augmentation**:Create an image generator for rotating, shifting, rescaling, and flipping to enhance model generalization.
4. **Build CNN Model**:Design a convolutional neural network with convolutional layers, max-pooling, and fully connected layers; compile the model.
5. **Train Model**:Split the dataset into training and testing sets, then train the model using the training data.
6. **Evaluate Performance**:Assess the model using the testing data, generating a classification report and confusion matrix.

### Program:
```Python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

my_data_dir = 'dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
os.listdir(train_path)
print(len(os.listdir(train_path+'/uninfected/')))
print(len(os.listdir(train_path+'/parasitized/')))
para_img= imread(train_path+'/parasitized/'+ os.listdir(train_path+'/parasitized')[0])
plt.imshow(para_img)
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
print(dim1)

sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
image_gen = ImageDataGenerator(rotation_range=20,width_shift_range=0.10, height_shift_range=0.10, 
                               rescale=1/255, shear_range=0.1, zoom_range=0.1, 
                               horizontal_flip=True,fill_mode='nearest')
model = models.Sequential()
model.add(keras.Input(shape=(image_shape)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
batch_size = 16
train_image_gen = image_gen.flow_from_directory(train_path,target_size=image_shape[:2],
                                                color_mode='rgb',batch_size=batch_size,class_mode='binary')
test_image_gen = image_gen.flow_from_directory(test_path,target_size=image_shape[:2],
                                               color_mode='rgb',batch_size=batch_size,class_mode='binary',shuffle=False)
results = model.fit(train_image_gen,epochs=4,validation_data=test_image_gen)
model.save('cell_model.h5')
losses = pd.DataFrame(model.history.history)
print("ROHIT JAIN D")
losses[['loss','val_loss']].plot()

import random
import tensorflow as tf
list_dir=["UnInfected","parasitized"]
dir_=(list_dir[1])
para_img= imread(train_path+ '/'+dir_+'/'+ os.listdir(train_path+'/'+dir_)[random.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(para_img))
img = tf.image.resize(img,(130,130))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred
    else "Un Infected")+"\nActual Value: "+str(dir_))
plt.axis("off")
print("ROHIT JAIN D 212222230120")
plt.imshow(img)
plt.show()

model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
test_image_gen.classes
predictions = pred_probabilities > 0.5
print("ROHIT JAIN D 212222230120")
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)
```
### Output:

##### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/fcabd1a3-955b-4b6d-837d-d3c22d60698c)
##### Classification Report
![image](https://github.com/user-attachments/assets/d2c6cc3a-dc96-47d3-a55e-91fd45fd8944)
##### Confusion Matrix
![image](https://github.com/user-attachments/assets/bd384fc7-a794-4506-8b16-60cf0765cc54)
##### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/68374572-2654-44ce-819a-812a7e562174)
### Result:
Thus a deep neural network for Malaria infected cell recognition and to analyze the performance is created using tensorflow.

**Developed By: ROHIT JAIN D - 212222230120**
