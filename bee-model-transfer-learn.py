from genericpath import exists
import os
from os import getcwd, walk
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.spines as spines
import seaborn as sns
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import tensorflow.keras.callbacks as callbacks

CUDA_VISIBLE_DEVICES=""

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

#print(tf.config.experimental.list_physical_devices('GPU') )

#print (tf.test.gpu_device_name())

DATASET_PATH = "./kaggle_bee_vs_wasp"

label_csv = os.path.join(DATASET_PATH, 'labels.csv')
label_df = pd.read_csv(label_csv)
label_df.head(10)

print('Number of null values')
print(label_df.isnull().sum())

def set_train_type(row):
    if row['is_validation'] == 0 and row['is_final_validation'] == 0:
        return 'train'
    if row['is_validation'] == 1:
        return 'validation'
    return 'test'

label_df['type'] = label_df.apply(set_train_type, axis=1)
print('Number values of each type')
print(label_df['type'].value_counts())

""" insect_cat_counts = label_df["label"].value_counts()

plt.figure(figsize=(7,7))
g = sns.barplot(x=insect_cat_counts.index, y =insect_cat_counts, color='salmon')
g.set(title='Number of each category', ylim=[0,6000], yticks=[], ylabel='')
sns.despine(top=True, right=True, left=True, bottom=False)

for i in range(4):
    plt.text(x=i-0.12, y=insect_cat_counts[i] + 150, s=insect_cat_counts[i])
plt.show() """

""" plt.figure(figsize=(7,7))
g = sns.countplot(x='label', hue='type', data=label_df, palette="pastel")
g.set(xlabel='', ylabel='', title="Test/Validation/Test numbers of each category")
sns.despine()
plt.show() """

""" plt.figure(figsize=(7,7))
g = sns.countplot(x='label', hue='photo_quality', data=label_df, palette="pastel")
g.set(xlabel='', ylabel='', title="Image quality of each category")
sns.despine()
plt.show() """

def display_img(row, pos):
    #Because path use back slash which is 
    #not compatible for both windows nor linux environment
    #we will first replace back slash with forward slash
    fn = row['path'].replace('\\', os.sep)
    fn = os.path.join(DATASET_PATH, fn)
    #Read image from path
    img = cv2.imread(fn)
    #Resize all images with the same size
    img = cv2.resize(img, (128, 128))
    #Set RGB color for image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Display image, and set title
    plt.subplot(4, 5, pos)
    plt.imshow(img)
    plt.title(row['label'])
    #Remove ticks
    plt.xticks([])
    plt.yticks([])

label_df['path'] = label_df['path'].str.replace('\\', os.sep)
label_df['path'].head()

train_df = label_df[label_df['type'] == 'train']
valid_df = label_df[label_df['type'] == 'validation']
test_df = label_df[label_df['type'] == 'test']

TARGET_SIZE = (256, 256)
SEED = 42

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

train_datagen = datagen.flow_from_dataframe(train_df, 
                                            directory=DATASET_PATH, 
                                            x_col='path', 
                                            y_col='label',
                                            target_size=TARGET_SIZE,
                                            seed=42
                                           ) 

valid_datagen = datagen.flow_from_dataframe(valid_df,
                                            directory=DATASET_PATH, 
                                            x_col='path', 
                                            y_col='label',
                                            target_size=TARGET_SIZE,
                                            seed=42
                                           ) 

test_datagen = datagen.flow_from_dataframe(test_df, 
                                           directory=DATASET_PATH, 
                                           x_col='path', 
                                           y_col='label',
                                           target_size=TARGET_SIZE,
                                           seed=42
                                           ) 

n_class = len(label_df['label'].unique())

from tensorflow.keras.initializers import RandomNormal, Constant

model = models.Sequential([
    # Block 1
    layers.Conv2D(128, 3, padding='same', kernel_regularizer=keras.regularizers.L2(0.001), input_shape=(256,256,3)),
    layers.Conv2D(128, 3, padding='same', kernel_regularizer=keras.regularizers.L2(0.001)),
    layers.BatchNormalization(momentum=0.6, 
                              epsilon=0.005, 
                              beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                              gamma_initializer=Constant(value=0.9)
                             ),    
    layers.Activation('relu'),
    layers.MaxPooling2D(3),
    layers.Dropout(0.3),
    
    # Block 2
    layers.Conv2D(128, 3, padding='same', kernel_regularizer=keras.regularizers.L2(0.001)),
    layers.Conv2D(128, 3, padding='same', kernel_regularizer=keras.regularizers.L2(0.001)),
    layers.BatchNormalization(momentum=0.6, 
                              epsilon=0.005, 
                              beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                              gamma_initializer=Constant(value=0.9)
                             ),
    layers.Activation('relu'),
    layers.MaxPooling2D(3),
    layers.Dropout(0.3),
    
    # Block 3
    layers.Conv2D(256, 3, padding='same', kernel_regularizer=keras.regularizers.L2(0.001)),
    layers.Conv2D(256, 3, padding='same', kernel_regularizer=keras.regularizers.L2(0.001)),
    layers.BatchNormalization(momentum=0.6, 
                              epsilon=0.005, 
                              beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                              gamma_initializer=Constant(value=0.9)
                             ), 
    layers.Activation('relu'),
    layers.MaxPooling2D(3),
    layers.Dropout(0.3),
    
    # Block 4
    layers.Conv2D(512, 3, padding='same', kernel_regularizer=keras.regularizers.L2(0.001)),
    layers.Conv2D(512, 3, padding='same', kernel_regularizer=keras.regularizers.L2(0.001)),
    layers.BatchNormalization(momentum=0.6, 
                              epsilon=0.005, 
                              beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
                              gamma_initializer=Constant(value=0.9)
                             ), 
    layers.Activation('relu'),
    layers.MaxPooling2D(3),
    layers.Dropout(0.3),

    # Block 5
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.L2(0.001)),
    layers.Dense(n_class, activation='softmax')
])
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss=losses.BinaryCrossentropy(),
              metrics=['accuracy']
             )
early_stopping = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(train_datagen, 
          validation_data=valid_datagen, 
          callbacks=[early_stopping],
          batch_size=32,
          epochs=50,
         )