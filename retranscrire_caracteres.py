from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import numpy as np
import chess
import chess.pgn
import difflib
from test_chess import conversion_piece, conversion_piece_inverse
import Levenshtein

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


total_coups_suggeres = 0



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


import re




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


# Classe pour permettre une intervention de l'utilisateur pour la correction des coups
class InteractionUser:
    def __init__(self):
        self.interventions_utilisateur = 0

    def validation_user(self, coup):
        validation = input('Le coup est-il correct ? (Y/N): ').strip().lower()
        if validation == 'y':
            coup_valide = coup
        else:
            coup_valide = str(input('Saisir le coup correct: ').strip())
            coup_valide = conversion_piece(coup_valide)
            self.interventions_utilisateur += 1
        return coup_valide

    def get_interventions(self):
        return self.interventions_utilisateur

# Déclaration de variable pour le calcul de taux d'intervention
coup_traite_correct = 0
total_coups_traites = 0

def post_traitement(dict_caractere_joueur, jeu, intervention):
    '''
    Cette fonction permet d'effectuer un post-traitement des coups
    initialement prédits.
    
    Paramètres :
    dict_caractere_joueur (dict) : Dictionnaire contenant les caractères à prédire sous forme d'images.
    jeu (chess.Board) : Instance de l'échiquier pour vérifier la légalité des coups.
    intervention (InteractionUser): Instance de la classe InteractionUser qui permet une intervention de l'utilisateur
                                    pour la correction des coups.
    
    Retourne :
    coup_corrige (str) : Chaîne représentant les caractères détectés et convertis en coup, corrigé si nécessaire.
    '''

    global coup_traite_correct, total_coups_traites

    def obtenir_top_prediction(res, n=5):
        """Retourne les indices des n prédictions avec les probas les plus élévées."""
        indices = np.argsort(res)[0][-n:]
        #print(f"indices {indices}")
        return indices
    

    coup = ""
    top_predictions = []

    for caractere in dict_caractere_joueur:
        if caractere is None:
            print("Erreur : caractère non défini.")
            return None

        if not isinstance(caractere, np.ndarray):
            print("Erreur : le caractère n'est pas une image valide.")
            return None

        try:
            # Redimensionner et prétraiter l'image
            image_to_predict = np.expand_dims(cv.resize(caractere, img_size), axis=0) / 255.0
        except cv.error as e:
            print(f"Erreur lors du redimensionnement de l'image: {e}")
            return None

        res = model.predict(image_to_predict, verbose = 0)
        #print(f"probas {res}")

        # Obtenir les trois meilleures prédictions
        top_preds = obtenir_top_prediction(res)
        top_predictions.append(top_preds)

        # Utiliser la meilleure prédiction pour le coup initial
        top_prediction = next(cle for cle, valeur in classes.items() if valeur == top_preds[-1])
        coup += conversion[top_prediction]

    #print(f"top prediction {top_predictions}")
    print(f"coup initial prédit: {coup}")

    if len(coup) != 2:
        # Conversion en notation anglaise
        coup = conversion_piece(coup)
    else:
        coup = coup.lower()
        #print(f"coup minuscule : {coup}")
        # Remplacer tout coup contenant "o" par "o-o"
        if 'o' in coup:
            coup = "O-O"
            coup_traite_correct +=1
            total_coups_traites +=1
        
        


     # Trouver la Liste des coups possibles
    coups_possibles = list(map(jeu.san, jeu.legal_moves))
    #print(f"coups possibles: {list(map(conversion_piece_inverse,coups_possibles))}")
    
    

    # Vérifier si le coup prédit est parmi les coups possibles
    if coup in coups_possibles:
        #coup_valide = intervention.validation_user(coup)
        jeu.push_san(coup)
        return conversion_piece_inverse(coup)
    

    # Sinon
    
    # Trouver les trois meilleurs coups candidats
    candidats = []
    for i in range(len(coup)):
        for top_pred in top_predictions[i]:
            cle = next((cle for cle, valeur in classes.items() if valeur == top_pred), None)
            if cle is not None:
                candidat = coup[:i] + conversion[cle] + coup[i+1:]
                candidats.append(conversion_piece(candidat))
    
    #print(f"candidats: {list(map(conversion_piece_inverse,candidats))}")

    # Calcul de la similarité basée sur la distance de Levenshtein
    similarites = []
    for candidat in candidats:
        for coup2 in coups_possibles:
            # Calcul de la distance de Levenshtein
            distance = Levenshtein.distance(candidat, coup2)
            # Calcul de la similarité (la distance inversée)
            max_len = max(len(candidat), len(coup2))
            similarity = 1 - (distance / max_len)
            similarites.append((similarity, coup2))

    # Tri des similarités en ordre décroissant
    similarites.sort(reverse=True, key=lambda x: x[0])

    # Sélection du coup suggéré
    similarite, coup_suggere = similarites[0] if similarites else None

    print(f"coup suggéré : {conversion_piece_inverse(coup_suggere)}")
    #print(f"similarité : {similarite}")
    total_coups_traites +=1
    
    # Demander la validation de l'utilisateur pour le coup suggéré
    coup_valide = intervention.validation_user(coup_suggere)
    if coup_valide == coup_suggere:
        coup_traite_correct +=1
    if coup_valide in coups_possibles:
        jeu.push_san(coup_valide)
        return conversion_piece_inverse(coup_valide)

    # Retourner None si aucun coup valide n'est trouvé
    return None


# Calcul de la proportion des coups suggérés corrects
def proportion_coups_traite_correct():
    if total_coups_traites > 0:
        print(f"coups traités corrects : {coup_traite_correct}")
        print(f"coups totaux traités {total_coups_traites}")
        proportion = (coup_traite_correct / total_coups_traites) * 100
        return proportion
    else:
        return 0




