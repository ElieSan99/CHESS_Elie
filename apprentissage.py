from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras.layers import RandomFlip
from keras.layers import RandomRotation
from keras.layers import RandomTranslation
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os
from zipfile import ZipFile
from keras.optimizers import Adam
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau  # Importez le callback ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import seaborn as sns



import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


PATH_DATASET = 'new_dataset'
size = 128 # le dataset non compressé est constitué d'images 128x128 !!!??
img_size = (size, size)
batch_size = 32
nbClasses = 27

classes = {'egal':0,
           'un':1,
           'deux':2,
           'trois':3,
           'quatre':4,
           'cinq':5,
           'six':6,
           'sept':7,
           'huit':8,
           'a':9,
           'b':10,
           'cmin':11,
           'dmin':12,
           'e':13,
           'fmin':14,
           'g':15,
           'h':16,
           'C':17,
           'F':18,
           'T':19,
           'R':20,
           'D':21,
           'diese':22,
           'plus':23,
           'x':24,
           'tiret':25,
           'o':26}

train_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(PATH_DATASET +'\\train', classes = classes,
                                                target_size=img_size, batch_size=batch_size, class_mode='categorical',color_mode='grayscale')

val_datagen = ImageDataGenerator(rescale=1./255)

val_set = val_datagen.flow_from_directory(PATH_DATASET+'\\val', classes = classes,
                                          target_size=img_size, batch_size=batch_size, class_mode='categorical',color_mode='grayscale')

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(PATH_DATASET+'\\test', classes = classes,
                                          target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False,color_mode='grayscale')

# Définition du callback ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.001)

model = Sequential()

# Couche Lambda pour convertir les images en niveaux de gris
model.add(tf.keras.layers.InputLayer(input_shape=(size, size, 1)))
# Couche de convolution 1
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Couche de convolution 2
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Couche de convolution 3
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Couche entièrement connectée
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

# Couche de sortie
model.add(Dense(nbClasses, activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

history = model.fit(training_set, validation_data=val_set,
                    steps_per_epoch=training_set.n // training_set.batch_size,
                    validation_steps=val_set.n // val_set.batch_size, epochs=30, verbose=1,
                    callbacks=[reduce_lr, early_stopping])  # Ajout du callback ReduceLROnPlateau

model.save("poids_cnn")

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('precision.png', dpi=300)
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('loss.png', dpi=300)
plt.show()

print("\nScores sur la base de test")
score = model.evaluate(test_set)
print("erreur = ", score[0])
print("justesse = ", score[1]*100, "%")

# Prédire les classes pour les données de test
y_pred = model.predict(test_set)
y_pred_classes = np.argmax(y_pred, axis=1)
print(f"classes prédites {np.unique(y_pred_classes)}")
y_true = test_set.classes
print(f"classes réelles {np.unique(y_true)}")

# Calculer la matrice de confusion
confusion_mtx=confusion_matrix(y_true, y_pred_classes, labels=range(nbClasses))

# Conversion du dictionnaire de classes
inv_classes = {v: k for k, v in classes.items()}

# Calcul du pourcentage de la matrice de confusion
class_totals = confusion_mtx.sum(axis=1)
confusion_mtx_percent = (confusion_mtx / class_totals[:, np.newaxis]) * 100

# Affichage de la matrice de confusion avec une heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(confusion_mtx_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=[inv_classes[i] for i in range(nbClasses)],
            yticklabels=[inv_classes[i] for i in range(nbClasses)])
plt.ylabel('Vrai label')
plt.xlabel('Label prédit')
plt.title('Matrice de Confusion en %')
plt.savefig('matrice_confusion.png', dpi=300)
plt.show()

"""
y_pred_probs = model.predict(test_set)
y_pred_probs = model.predict(test_set)  # Probabilités pour chaque classe
y_pred_classes = np.argmax(y_pred_probs, axis=1)  # Classes prédictes
y_true = test_set.classes  # Classes réelles

for i in range(5):
    print(f"Image {i+1}:")
    print(f"  Probabilités prédites : {y_pred_probs[i]}")
    print(f"  Classe prédite : {y_pred_classes[i]}")
    print(f"  Classe réelle : {y_true[i]}")
"""
"""
# Prédire les classes pour les données de test
y_pred = model.predict(test_set)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_set.classes

# Calculer la matrice de confusion
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
"""
"""# Calculer la précision et le rappel
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print("Matrice de confusion :")
print(confusion_mtx)
print("Précision :", precision)
print("Rappel :", recall)
print("F1 :" , f1)"""
