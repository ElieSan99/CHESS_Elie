from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import numpy as np
import chess
import chess.pgn
import difflib
from test_chess import conversion_piece, conversion_piece_inverse

#list_caract = ["a","b","C","5","cmin","D","2","#","dmin","e","=","F","fmin","g","h","8","o","+","4","R","7","6","T","-","3","1","x"]

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

conversion = {'egal':'=',
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

def get_key_from_value(dico, valeur):
    """
    Retourne la première clé associée à une valeur spécifique dans le dictionnaire.

    Paramètres :
    dico (dict) : Dictionnaire dans lequel rechercher la clé.
    valeur : Valeur dont on recherche la clé associée.

    Retourne :
    cle (str) : La première clé trouvée correspondant à la valeur spécifiée.
                Retourne None si aucune clé n'est trouvée pour cette valeur.
    """
    for cle, val in dico.items():
        if val == valeur:
            return cle
    return None  # Retourne None si aucune clé n'est trouvée pour la valeur donnée


size = 128
img_size = (size, size) # verifier taille image (128x128 dans base apprentissage ??)

model = keras.models.load_model("poids_cnn", custom_objects=None, compile=True, options=None)


def retranscrire_caractere(caractere) :
    '''
    Cette fonction permet de redimensionner et prétraiter l'image,
    puis de calculer quel caractère est présent.
    
    Paramètre :
    image (array) --> Liste de liste, comprenant les valeurs de l'image
    contenant le caractère recherché.
    
    Retourne :
    predicted_class (str) --> String comprenant la valeur du caractère détecté.
    '''

    #caractere = cv.merge([caractere, caractere, caractere])
    image_to_predict = np.expand_dims(cv.resize(caractere, img_size), axis=0) / 255.0

    
    res = model.predict(image_to_predict)

    #print(res)
    predicted_class_index = np.argmax(res)
    #print('index', predicted_class_index)

    # Supprimer la probabilité maximale de res
    #res = np.delete(res, predicted_class_index)

    for cle, valeur in classes.items():
        if valeur == predicted_class_index:
            return(conversion[cle])
    return '?'


#### A voir 


def retranscrire_caractere2(dict_caractere_joueur, jeu):
    '''
    Cette fonction permet de redimensionner et prétraiter l'image,
    puis de calculer quel caractère est présent.
    
    Paramètres :
    dict_caractere_joueur (dict) : Dictionnaire contenant les caractères à prédire sous forme d'images.
    jeu (chess.Board) : Instance de l'échiquier pour vérifier la légalité des coups.
    
    Retourne :
    coup_corrige (str) : Chaîne représentant les caractères détectés et convertis en coup, corrigé si nécessaire.
    '''

    def obtenir_top_prediction(res, n=3):
        """Retourne les indices des n prédictions avec les probas les plus élévées."""
        indices = np.argsort(res)[0][-n:]
        print(f"indices {indices}")
        return indices
    
    def validation_user(coup):
        """Retourne le coup validé par l'utilisateur"""

        validation = input('Le coup est il correct ? (Y/N): ').strip().lower()
        if validation=='y':
            coup_valide = coup
        else:
            coup_valide = str(input('Saisir le coup correct: ').strip())
            coup_valide = conversion_piece(coup_valide)

        return coup_valide

    coup = ""
    top_predictions = []

    for caractere in dict_caractere_joueur:

        # Afficher le car
        #cv.imshow('car', caractere)
        #cv.waitKey(0)
        # Redimensionner et prétraiter l'image
        image_to_predict = np.expand_dims(cv.resize(caractere, img_size), axis=0) / 255.0
        res = model.predict(image_to_predict)

        print(f"probas {res}")

        # Obtenir les trois meilleures prédictions
        top_preds = obtenir_top_prediction(res)
        top_predictions.append(top_preds)

        # Utiliser la meilleure prédiction pour le coup initial
        top_prediction = next(cle for cle, valeur in classes.items() if valeur == top_preds[-1])
        coup += conversion[top_prediction]

    print(f"top prediction {top_predictions}")
    print(f"coup initial prédit: {coup}")

    if len(coup) != 2:
        # Conversion en notation anglaise
        coup = conversion_piece(coup)
    else:
        coup = coup.lower()
        print(f"coup minuscule : {coup}")

    
    # Remplacer tout coup contenant "o" par "o-o"
    if 'o' in coup:
        coup = "O-O"

    # Liste des coups possibles
    coups_possibles = list(map(jeu.san, jeu.legal_moves))

    print(f"coups possibles: {coups_possibles}")

    coup_corrige = coup
    nb_correction = 0
    max_tentatives = 3  # Maximum de tentatives de correction

    while nb_correction < max_tentatives:
        print(f"Tentative de correction {nb_correction + 1}: {coup_corrige}")
        
        if coup_corrige in coups_possibles:
            coup_corrige = validation_user(coup_corrige)
            jeu.push_san(coup_corrige)
            return conversion_piece_inverse(coup_corrige)
        
        # Sinon
        
        # Trouver le coup le plus similaire parmi les coups possibles
        similarites = [(difflib.SequenceMatcher(None, coup_corrige, coup2).ratio(), coup2) for coup2 in coups_possibles]
        similarites.sort(reverse=True, key=lambda x: x[0])
        coup_suggere = similarites[0][1] if similarites else coup_corrige
        print(f"coup suggéré : {coup_suggere}")
        

        # Trouver et corriger le caractère différent
        for i, (char1, char2) in enumerate(zip(coup_corrige, coup_suggere)):
            if char1 != char2:
                print(f"char1 {char1}")
                # Utiliser la prochaine meilleure prédiction pour le caractère différent
                if nb_correction > 1:
                    cle1 = get_key_from_value(conversion,char1)
                else:
                    cle1 = get_key_from_value(conversion,conversion_piece_inverse(char1))

                print(f"cle1 {cle1}")
                current_best_index = np.where(top_predictions[i] == classes[cle1])[0][0]
                next_best_index = current_best_index - 1  # Prédiction suivante la plus probable
                if next_best_index >= 0:
                    next_best_prediction = next(cle for cle, valeur in classes.items() if valeur == top_predictions[i][next_best_index])
                    coup_corrige = coup_corrige[:i] + conversion[next_best_prediction] + coup_corrige[i+1:]
                break  # Sortir de la boucle une fois le caractère corrigé

        nb_correction += 1

    # Si aucune correction n'a fonctionné, retourner le coup suggéré
    coup_suggere = validation_user(coup_suggere)
    jeu.push_san(coup_suggere)
    return conversion_piece_inverse(coup_suggere)
