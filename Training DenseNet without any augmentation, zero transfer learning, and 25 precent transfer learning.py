#importing necessary libraries

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))  # gpu check
import os
import numpy as np
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import pandas as pd


#defining class names and preparing training and validation dataset

class_names = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
trainset = [class_names[0],class_names[1],class_names[2],class_names[3],class_names[4],class_names[5],class_names[6],class_names[7]]

#training dataset
for i in range(len(class_names)):
    trainset[i] = os.path.join('train/',class_names[i])
print("Size of each class of Training Set: ")
for i in range(len(trainset)):
    print(class_names[i],' has ',len(os.listdir(trainset[i])), 'instances.')
    
#Validation Dataset
validationset = [class_names[0],class_names[1],class_names[2],class_names[3],class_names[4],class_names[5],class_names[6],class_names[7]]
for i in range(len(class_names)):
    validationset[i] = os.path.join('test/',class_names[i])
print("\nSize of each class of Validation Set: ")
for i in range(len(validationset)):
    print(class_names[i],' has ',len(os.listdir(validationset[i])), 'instances.')
    
#without any augmentaion

train_datagen = ImageDataGenerator(rescale = 1./255,)
                                
validation_datagen = ImageDataGenerator(rescale = 1./255)

BATCH_SIZE = 16
IMG_HEIGHT = 224
IMG_WIDTH = 224

#calculation of steps per epoch

total_train_images = 0;
for i in range(len(trainset)):
    total_train_images += len(os.listdir(trainset[i]))
    
STEPS_PER_EPOCH = np.ceil(total_train_images/BATCH_SIZE)
    
#image capturing using datagen

training_set = train_datagen.flow_from_directory(directory = 'train', #directory=str(data_dir)
                                                 target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical')


validation_set = validation_datagen.flow_from_directory(directory = 'test',
                                                 target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical')

#adam optimizer definition
adam = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False) 

#DenseNet
base_model_d = tf.keras.applications.densenet.DenseNet201(
    include_top= False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

x_d = base_model_d.output
x_d = GlobalAveragePooling2D()(x_d)
x_d = Dense(1024, activation='sigmoid')(x_d)
x_d = Dense(1024, activation='sigmoid')(x_d)
predictions_d = Dense(8, activation='sigmoid')(x_d)
model_d = Model(inputs=base_model_d.input, outputs=predictions_d)

model_d.summary()

a = len(model_d.layers)/2
a = int(a)

for layer in model_d.layers[:a]:
    layer.trainable = False
for layer in model_d.layers[a:]:
    layer.trainable = True

model_d.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=adam,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC(),'accuracy','top_k_categorical_accuracy'])

history_d = model_d.fit(training_set,
                          steps_per_epoch = STEPS_PER_EPOCH, #total_training_images/batch_size
                          epochs = 30,
                          validation_data = validation_set)



#saving model parameters

desne_half_aug = pd.DataFrame(history_d.history)
hist_csv_file = 'desne_zero_aug.csv'
with open(hist_csv_file, mode='w') as f:
    desne_half_aug.to_csv(f)
    
    
#zero transfer learning

#Augmentation using keras

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range=40,
                                    width_shift_range =0.2,
                                    height_shift_range=0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
                                
validation_datagen = ImageDataGenerator(rescale = 1./255)

BATCH_SIZE = 16
IMG_HEIGHT = 224
IMG_WIDTH = 224

#calculation of steps per epoch

total_train_images = 0;
for i in range(len(trainset)):
    total_train_images += len(os.listdir(trainset[i]))
    
STEPS_PER_EPOCH = np.ceil(total_train_images/BATCH_SIZE)
    
#image capturing using datagen

training_set = train_datagen.flow_from_directory(directory = 'train', #directory=str(data_dir)
                                                  target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                  batch_size = BATCH_SIZE,
                                                  class_mode = 'categorical')


validation_set = validation_datagen.flow_from_directory(directory = 'test',
                                                  target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                  batch_size = BATCH_SIZE,
                                                  class_mode = 'categorical')

#model definitions. densenet, vggnet, mobilenet, inception and xception is used here
adam = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False) 

#DenseNet
base_model_d = tf.keras.applications.densenet.DenseNet201(
    include_top= False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

x_d = base_model_d.output
x_d = GlobalAveragePooling2D()(x_d)
x_d = Dense(1024, activation='sigmoid')(x_d)
x_d = Dense(1024, activation='sigmoid')(x_d)
predictions_d = Dense(8, activation='sigmoid')(x_d)
model_d = Model(inputs=base_model_d.input, outputs=predictions_d)

model_d.summary()

a = len(model_d.layers)/2
a = int(a)

for layer in model_d.layers[0:]:
    layer.trainable = False
   
model_d.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=adam,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC(),'accuracy','top_k_categorical_accuracy'])

history_d = model_d.fit(training_set,
                          steps_per_epoch = STEPS_PER_EPOCH, #total_training_images/batch_size
                          epochs = 30,
                          validation_data = validation_set)



#saving model parameters

desne_half_aug = pd.DataFrame(history_d.history)
hist_csv_file = 'desne_zero_transfer learning.csv'
with open(hist_csv_file, mode='w') as f:
    desne_half_aug.to_csv(f)

#25% transfer learning    
    
#DenseNet
base_model_d = tf.keras.applications.densenet.DenseNet201(
    include_top= False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

x_d = base_model_d.output
x_d = GlobalAveragePooling2D()(x_d)
x_d = Dense(1024, activation='sigmoid')(x_d)
x_d = Dense(1024, activation='sigmoid')(x_d)
predictions_d = Dense(8, activation='sigmoid')(x_d)
model_d = Model(inputs=base_model_d.input, outputs=predictions_d)

model_d.summary()

a = len(model_d.layers)/3
a = int(a)
a = 2*a
for layer in model_d.layers[:a]:
    layer.trainable = False
for layer in model_d.layers[a:]:
    layer.trainable = True
   
model_d.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=adam,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC(),'accuracy','top_k_categorical_accuracy'])

history_d = model_d.fit(training_set,
                          steps_per_epoch = STEPS_PER_EPOCH, #total_training_images/batch_size
                          epochs = 30,
                          validation_data = validation_set)
#saving model parameters

desne_half_aug = pd.DataFrame(history_d.history)
hist_csv_file = 'desne_25%_transfer learning.csv'
with open(hist_csv_file, mode='w') as f:
    desne_half_aug.to_csv(f)
