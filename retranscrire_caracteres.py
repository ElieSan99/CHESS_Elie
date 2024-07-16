from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import numpy as np

#list_caract = ["a","b","C","5","cmin","D","2","#","dmin","e","=","F","fmin","g","h","8","o","+","4","R","7","6","T","-","3","1","x"]

classes = {'=':0,
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

conversion = {'=':'=',
           'un':'1',
           'deux':'2',
           'trois':'3',
           'quatre':'4',
           'cinq':'5',
           'six':'6',
           'sept':'7',
           'huit':'8',
           'a':'a',
           'b':'b',
           'cmin':'c',
           'dmin':'d',
           'e':'e',
           'fmin':'f',
           'g':'g',
           'h':'h',
           'C':'C',
           'F':'F',
           'T':'T',
           'R':'R',
           'D':'D',
           'diese':'#',
           'plus':'+',
           'x':'x',
           'tiret':'-',
           'o':'o'}

size = 128
img_size = (size, size) # verifier taille image (128x128 dans base apprentissage ??)

model = keras.models.load_model("poids_03_06_2024", custom_objects=None, compile=True, options=None)


def retranscrire_caractere (caractere) :
    '''
    Cette fonction permet de redimensionner et prétraiter l'image,
    puis de calculer quel caractère est présent.
    
    Paramètre :
    image (array) --> Liste de liste, comprenant les valeurs de l'image
    contenant le caractère recherché.
    
    Retourne :
    predicted_class (str) --> String comprenant la valeur du caractère détecté.
    '''
    caractere = cv.merge([caractere, caractere, caractere])
    image_to_predict = np.expand_dims(cv.resize(caractere, img_size), axis=0) / 255.0
    res = model.predict(image_to_predict)
    predicted_class_index = np.argmax(res)
    for cle, valeur in classes.items():
        if valeur == predicted_class_index:
            return(conversion[cle])
    return '?'
