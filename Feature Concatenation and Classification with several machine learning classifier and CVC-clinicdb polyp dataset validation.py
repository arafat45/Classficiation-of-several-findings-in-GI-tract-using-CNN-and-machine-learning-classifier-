import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support
import pickle
from sklearn.metrics import ConfusionMatrixDisplay

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

SIZE = 224  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 

for directory_path in glob.glob("train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)
        
#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = [] 
for directory_path in glob.glob("test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#One hot encode y values for neural network. 
from tensorflow.keras.utils import to_categorical

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#polyps dataset

p_test_images = []
p_test_labels = [] 
for directory_path in glob.glob("CVC-clinicDB/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        p_test_images.append(img)
        p_test_labels.append(fruit_label)


#Convert lists to arrays                
p_test_images = np.array(p_test_images)
p_test_labels = np.array(p_test_labels)

#Encode labels from text to integers.
le = preprocessing.LabelEncoder()
le.fit(p_test_labels)
p_test_labels_encoded = le.transform(p_test_labels)



#Split data into test and train datasets (already split but assigning to meaningful convention)
x_test_p, y_test_p = p_test_images, p_test_labels_encoded
# Normalize pixel values to between 0 and 1

x_test_p = x_test_p/255.0

#One hot encode y values for neural network. 
y_test_p_one_hot = to_categorical(y_test_p)

dense_model_full=tf.keras.models.load_model('dense-weights-improvement-22-0.93.h5')
dense_model= tf.keras.Model(dense_model_full.input,
              dense_model_full.layers[-2].output)


inc_model_full=tf.keras.models.load_model('inception-weights-improvement-29-0.92.h5')
inc_model= tf.keras.Model(inc_model_full.input,
              inc_model_full.layers[-2].output)

xcp_model_full=tf.keras.models.load_model('xception-weights-improvement-11-0.92.h5')
xcp_model= tf.keras.Model(xcp_model_full.input,
              xcp_model_full.layers[-2].output)

vgg_model_full=tf.keras.models.load_model('vgg-weights-improvement-15-0.87.h5')
vgg_model= tf.keras.Model(vgg_model_full.input,
              vgg_model_full.layers[-2].output)

for layer in dense_model.layers:
	layer.trainable = False
    
for layer in inc_model.layers:
	layer.trainable = False
    
for layer in xcp_model.layers:
	layer.trainable = False

for layer in vgg_model.layers:
	layer.trainable = False

#Now, let us use features from convolutional network for RF
feature_extractor_dense=dense_model.predict(x_train)
features_dense = feature_extractor_dense.reshape(feature_extractor_dense.shape[0], -1)
print('dense done')

feature_extractor_inc=inc_model.predict(x_train)
features_inc = feature_extractor_inc.reshape(feature_extractor_inc.shape[0], -1)
print('inception done done')

feature_extractor_xcp=xcp_model.predict(x_train)
features_xcp = feature_extractor_xcp.reshape(feature_extractor_xcp.shape[0], -1)
print('xception done done')

feature_extractor_vgg=vgg_model.predict(x_train)
features_vgg = feature_extractor_vgg.reshape(feature_extractor_xcp.shape[0], -1)
print('vgg done done')


features_dense = np.array(features_dense)
features_inc = np.array(features_inc)
features_xcp = np.array(features_xcp)
features_vgg = np.array(features_vgg)

#Send test data through same feature extractor process
X_test_feature_dense = dense_model.predict(x_test)
X_test_features_dense = X_test_feature_dense.reshape(X_test_feature_dense.shape[0], -1)

X_test_feature_inc = inc_model.predict(x_test)
X_test_features_inc = X_test_feature_inc.reshape(X_test_feature_inc.shape[0], -1)

X_test_feature_xcp = xcp_model.predict(x_test)
X_test_features_xcp = X_test_feature_xcp.reshape(X_test_feature_xcp.shape[0], -1)

X_test_feature_vgg = vgg_model.predict(x_test)
X_test_features_vgg = X_test_feature_vgg.reshape(X_test_feature_vgg.shape[0], -1)

X_test_features_dense = np.array(X_test_features_dense)
X_test_features_inc = np.array(X_test_features_inc)
X_test_features_xcp = np.array(X_test_features_xcp)
X_test_features_vgg = np.array(X_test_features_vgg)

#polyps dataset in feature extraction
X_test_feature_dense_p = dense_model.predict(x_test_p)
X_test_features_dense_p = X_test_feature_dense_p.reshape(X_test_feature_dense_p.shape[0], -1)

X_test_feature_inc_p = inc_model.predict(x_test_p)
X_test_features_inc_p = X_test_feature_inc_p.reshape(X_test_feature_inc_p.shape[0], -1)

X_test_feature_xcp_p = xcp_model.predict(x_test_p)
X_test_features_xcp_p = X_test_feature_xcp_p.reshape(X_test_feature_xcp_p.shape[0], -1)

X_test_feature_vgg_p = vgg_model.predict(x_test_p)
X_test_features_vgg_p = X_test_feature_vgg_p.reshape(X_test_feature_vgg_p.shape[0], -1)

X_test_features_dense_p = np.array(X_test_features_dense_p)
X_test_features_inc_p = np.array(X_test_features_inc_p)
X_test_features_xcp_p = np.array(X_test_features_xcp_p)
X_test_features_vgg_p = np.array(X_test_features_vgg_p)

#features concatenatenation inceptionNet+denseNet, inceptionNet+denseNet+XceptionNet, inceptionNet+denseNet+XceptionNet+VggNet

series_id = np.concatenate((features_inc,features_dense), axis=1)
series_id_test = np.concatenate((X_test_features_inc,X_test_features_dense), axis=1)

series_idx = np.concatenate((features_inc,features_dense,features_xcp), axis=1)
series_idx_test = np.concatenate((X_test_features_inc,X_test_features_dense,X_test_features_xcp), axis=1)
series_idx_test_p = np.concatenate((X_test_features_inc_p,X_test_features_dense_p,X_test_features_xcp_p), axis=1)

series_idxv = np.concatenate((features_inc,features_dense,features_xcp,features_vgg), axis=1)
series_idxv_test = np.concatenate((X_test_features_inc,X_test_features_dense,X_test_features_xcp,X_test_features_vgg), axis=1)

#RANDOM FOREST

model = RandomForestClassifier(n_estimators = 50, random_state = 42)
# Train the model on training data
model.fit(series_id, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_id_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("RF Series id Accuracy  = ", metrics.accuracy_score(test_labels, prediction))

# Train the model on training data
model.fit(series_idx, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idx_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("RF Series idx Accuracy  = ", metrics.accuracy_score(test_labels, prediction))
print ("RF precision recall f1score = ", precision_recall_fscore_support(test_labels, prediction, average='weighted'))

# Train the model on training data
model.fit(series_idxv, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idxv_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("RF Series idxv Accuracy  = ", metrics.accuracy_score(test_labels, prediction))

#Xgb boost

model = xgb.XGBClassifier()
# Train the model on training data
model.fit(series_id, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_id_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("Xgb Series id Accuracy  = ", metrics.accuracy_score(test_labels, prediction))

# Train the model on training data
model.fit(series_idx, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idx_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("Xgb Series idx Accuracy  = ", metrics.accuracy_score(test_labels, prediction))
print ("xgb precision recall f1score = ", precision_recall_fscore_support(test_labels, prediction, average='weighted'))

# Train the model on training data
model.fit(series_idxv, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idxv_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("Xgb Series idxv Accuracy  = ", metrics.accuracy_score(test_labels, prediction))


#SVM

from sklearn import svm # import SVM
model = svm.SVC(C=10, gamma=0.1)
# Train the model on training data
model.fit(series_id, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_id_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("SVM Series id Accuracy  = ", metrics.accuracy_score(test_labels, prediction))

# Train the model on training data
model.fit(series_idx, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idx_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("SVM Series idx Accuracy  = ", metrics.accuracy_score(test_labels, prediction))
print ("svm precision recall f1score = ", precision_recall_fscore_support(test_labels, prediction, average='weighted'))

# Train the model on training data
model.fit(series_idxv, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idxv_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("SVM Series idxv Accuracy  = ", metrics.accuracy_score(test_labels, prediction))

#LogisticRegression 

from sklearn.linear_model import LogisticRegression 
model= LogisticRegression(solver='liblinear')
# Train the model on training data
model.fit(series_id, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_id_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("LogisticRegression Series id Accuracy  = ", metrics.accuracy_score(test_labels, prediction))


# Train the model on training data
model.fit(series_idx, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idx_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("LogisticRegression Series idx Accuracy  = ", metrics.accuracy_score(test_labels, prediction))
print ("log precision recall f1score = ", precision_recall_fscore_support(test_labels, prediction, average='weighted'))


# Train the model on training data
model.fit(series_idxv, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idxv_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("LogisticRegression Series idxv Accuracy  = ", metrics.accuracy_score(test_labels, prediction))

#naive_bayes

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
# Train the model on training data
model.fit(series_id, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_id_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("naive_bayes Series id Accuracy  = ", metrics.accuracy_score(test_labels, prediction))


# Train the model on training data
model.fit(series_idx, y_train) #For sklearn no one hot encoding
filename = 'naive_bayes_model.sav'
pickle.dump(model, open(filename, 'wb'))
#Now predict using the trained RF model. 
prediction = model.predict(series_idx_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("naive_bayes Series idx Accuracy  = ", metrics.accuracy_score(test_labels, prediction))
print ("naive_bayes recall f1score = ", precision_recall_fscore_support(test_labels, prediction, average='weighted'))

ConfusionMatrixDisplay.from_predictions(test_labels, prediction)
plt.show()

p = precision_recall_fscore_support(test_labels,prediction, average=None,labels=['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis','normal-cecum','normal-pylorus','normal-z-line','polyps','ulcerative-colitis'])

#testing polyps dataset
prediction = model.predict(series_idx_test_p)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("naive_bayes Series idx Accuracy for polyp dataset  = ", metrics.accuracy_score(p_test_labels, prediction))
print ("naive_bayes recall f1score for ployp dataset = ", precision_recall_fscore_support(p_test_labels, prediction, average='weighted'))

ConfusionMatrixDisplay.from_predictions(p_test_labels, prediction)
plt.show()

p_p = precision_recall_fscore_support(h_test_labels,prediction, average=None,labels=['polyps'])

# Train the model on training data
model.fit(series_idxv, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idxv_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("naive_bayes Series idxv Accuracy  = ", metrics.accuracy_score(test_labels, prediction))


#mlp

from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
# Train the model on training data
model.fit(series_id, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_id_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("mlp Series id Accuracy  = ", metrics.accuracy_score(test_labels, prediction))

# Train the model on training data
model.fit(series_idx, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idx_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("mlp Series idx Accuracy  = ", metrics.accuracy_score(test_labels, prediction))
print ("mlp recall f1score = ", precision_recall_fscore_support(test_labels, prediction, average='weighted'))
ConfusionMatrixDisplay.from_predictions(test_labels, prediction)
plt.show()

# Train the model on training data
model.fit(series_idxv, y_train) #For sklearn no one hot encoding
#Now predict using the trained RF model. 
prediction = model.predict(series_idxv_test)
#Inverse le transform to get original label back. 
prediction = le.inverse_transform(prediction)

print ("mlp Series idxv Accuracy  = ", metrics.accuracy_score(test_labels, prediction))








