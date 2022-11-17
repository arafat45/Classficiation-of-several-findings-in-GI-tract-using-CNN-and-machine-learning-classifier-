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
adam = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False) #

# #DenseNet
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
   
filepath="dense model/dense-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5" 
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_d.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=adam,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC(),'accuracy','top_k_categorical_accuracy'])

history_d = model_d.fit(training_set,
                          steps_per_epoch = STEPS_PER_EPOCH, #total_training_images/batch_size
                          epochs = 30,
                          validation_data = validation_set,
                          callbacks=callbacks_list)



#saving model parameters

desne_half_aug = pd.DataFrame(history_d.history)
hist_csv_file = 'desne.csv'
with open(hist_csv_file, mode='w') as f:
    desne_half_aug.to_csv(f)

#reset

base_model_r = tf.keras.applications.resnet.ResNet152(
    include_top= False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

x_r = base_model_r.output
x_r = GlobalAveragePooling2D()(x_r)
x_r = Dense(1024, activation='sigmoid')(x_r)
x_r = Dense(1024, activation='sigmoid')(x_r)
predictions_r = Dense(8, activation='sigmoid')(x_r)
model_r = Model(inputs=base_model_r.input, outputs=predictions_r)

model_r.summary()

a = len(model_r.layers)/2
a = int(a)

for layer in model_r.layers[:a]:
    layer.trainable = False
for layer in model_r.layers[a:]:
    layer.trainable = True
   
filepath=" rensenet model/rense-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5" 
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_r.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=adam,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC(),'accuracy','top_k_categorical_accuracy'])

history_r = model_r.fit(training_set,
                          steps_per_epoch = STEPS_PER_EPOCH, #total_training_images/batch_size
                          epochs = 30,
                          validation_data = validation_set,
                          callbacks=callbacks_list)

#saving model parameters

res_half_aug = pd.DataFrame(history_r.history)
hist_csv_file = 'resnet.csv'
with open(hist_csv_file, mode='w') as f:
        res_half_aug.to_csv(f)


#Vgg19

base_model_v = tf.keras.applications.vgg19.VGG19(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

x_v = base_model_v.output
x_v = GlobalAveragePooling2D()(x_v)
x_v = Dense(1024, activation='sigmoid')(x_v)
x_v = Dense(1024, activation='sigmoid')(x_v)
predictions_v= Dense(8, activation='sigmoid')(x_v)
model_v= Model(inputs=base_model_v.input, outputs=predictions_v)

model_v.summary()

a = len(model_v.layers)/2
a = int(a)

for layer in model_v.layers[:a]:
   layer.trainable = False
for layer in model_v.layers[a:]:
   layer.trainable = True
   
filepath="vgg model/vgg-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5" 
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_v.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=adam,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC(),'accuracy','top_k_categorical_accuracy'])

history_v = model_v.fit(training_set,
                         steps_per_epoch = STEPS_PER_EPOCH, #total_training_images/batch_size
                         epochs = 30,
                         validation_data = validation_set,
                         callbacks=callbacks_list)


#saving models and all the parameters

vgg_half_aug = pd.DataFrame(history_v.history)
hist_csv_file = 'vgg.csv'
with open(hist_csv_file, mode='w') as f:
    vgg_half_aug.to_csv(f)


#mobilenet

base_model_m = tf.keras.applications.MobileNetV3Large(
    input_shape=None,
    alpha=1.0,
    minimalistic = False,
    include_top = False,
    weights='imagenet',
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    include_preprocessing = True
)

x_m = base_model_m.output
x_m = GlobalAveragePooling2D()(x_m)
x_m = Dense(1024, activation='sigmoid')(x_m)
x_m = Dense(1024, activation='sigmoid')(x_m)
predictions_m= Dense(8, activation='sigmoid')(x_m)
model_m= Model(inputs=base_model_m.input, outputs=predictions_m)

model_m.summary()

a = len(model_m.layers)/2
a = int(a)

for layer in model_m.layers[:a]:
   layer.trainable = False
for layer in model_m.layers[a:]:
   layer.trainable = True
   
filepath="mobilenet model/mobile-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5" 
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_m.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=adam,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC(),'accuracy','top_k_categorical_accuracy'])

history_m = model_m.fit(training_set,
                         steps_per_epoch = STEPS_PER_EPOCH, #total_training_images/batch_size
                         epochs = 30,
                         validation_data = validation_set,
                         callbacks=callbacks_list)

#saving models and all the parameters

mobile_half_aug = pd.DataFrame(history_m.history)
hist_csv_file = 'mobile.csv'
with open(hist_csv_file, mode='w') as f:
    mobile_half_aug.to_csv(f)

#inception

base_model_i = tf.keras.applications.inception_v3.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

x_i = base_model_i.output
x_i = GlobalAveragePooling2D()(x_i)
x_i = Dense(1024, activation='sigmoid')(x_i)
x_i = Dense(1024, activation='sigmoid')(x_i)
predictions_i= Dense(8, activation='sigmoid')(x_i)
model_i= Model(inputs=base_model_i.input, outputs=predictions_i)

model_i.summary()

a = len(model_i.layers)/2
a = int(a)

for layer in model_i.layers[:a]:
   layer.trainable = False
for layer in model_i.layers[a:]:
   layer.trainable = True
   
filepath="inception model/inception-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5" 
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_i.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=adam,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC(),'accuracy','top_k_categorical_accuracy'])

history_i = model_i.fit(training_set,
                         steps_per_epoch = STEPS_PER_EPOCH, #total_training_images/batch_size
                         epochs = 30,
                         validation_data = validation_set,
                         callbacks=callbacks_list)


#saving models and all the parameters

inception_half_aug = pd.DataFrame(history_i.history)
hist_csv_file = 'inception.csv'
with open(hist_csv_file, mode='w') as f:
    inception_half_aug.to_csv(f)

#xception

base_model_x = tf.keras.applications.xception.Xception(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)

x_x = base_model_x.output
x_x = GlobalAveragePooling2D()(x_x)
x_x = Dense(1024, activation='sigmoid')(x_x)
x_x = Dense(1024, activation='sigmoid')(x_x)
predictions_x= Dense(8, activation='sigmoid')(x_x)
model_x= Model(inputs=base_model_x.input, outputs=predictions_x)

model_x.summary()

a = len(model_x.layers)/2
a = int(a)

for layer in model_x.layers[:a]:
   layer.trainable = False
for layer in model_x.layers[a:]:
   layer.trainable = True
   
filepath="xception model/xception-weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5" 
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model_x.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=adam,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.AUC(),'accuracy','top_k_categorical_accuracy'])

history_x = model_x.fit(training_set,
                         steps_per_epoch = STEPS_PER_EPOCH, #total_training_images/batch_size
                         epochs = 30,
                         validation_data = validation_set,
                         callbacks=callbacks_list)


#saving models and all the parameters

xception_half_aug = pd.DataFrame(history_x.history)
hist_csv_file = 'xception.csv'
with open(hist_csv_file, mode='w') as f:
    xception_half_aug.to_csv(f)
    

























































































