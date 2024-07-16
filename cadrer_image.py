'''Import des differentes fonctions necessaires'''
from detection_lignes import detection_ligne_haut, detection_ligne_droite,\
        detection_ligne_gauche, detection_ligne_bas

def cadrage_largeur(img) :
    '''
    Fonction qui appelle les fonctions detection_ligne_gauche et detection_ligne_droite
    pour obtenir les limites à garder de l'image.

    Paramètres :
    img (array) : Une liste de listes, ayant pour valeurs les pixels aux niveaux de gris
    de la photo initale.
    
    Retourne :
    img (array) : Une partie du paramètre en entrée, contenant uniquement les données
    comprises entre les deux limites détéctées.
    '''
    limite_droite = detection_ligne_droite(img)
    limite_gauche = detection_ligne_gauche(img)

    return img[:, limite_droite : limite_gauche]

def cadrage_hauteur(img) :
    '''
    Fonction qui appelle les fonctions detection_ligne_haut et detection_ligne_bas
    pour obtenir les limites à garder de l'image.

    Paramètres :
    img (array) : Une liste de listes, ayant pour valeurs les pixels aux niveaux de gris
    de la photo initale.

    Retourne :
    img (array) : Une partie du paramètre en entrée, contenant uniquement les données
    comprises entre les deux limites détéctées.
    '''

    limite_haut = detection_ligne_haut(img)
    print("limite_haut : ", limite_haut)
    limite_bas = detection_ligne_bas(img)
    print("limite_bas : ", limite_bas)
    return img[limite_haut-20: limite_bas,:]

def cadrage(img):
    '''
    Fonction qui appelle les fonctions cadrage_vertical et cadrage_horizontal pour les appliquer.

    Paramètres :
    img (array) : Une liste de listes, ayant pour valeurs les pixels aux niveaux de gris
    de la photo initale.

    Retourne :
    img_cadree (array) : Une partie du paramètre en entrée, contenant uniquement les données
    comprises entre les quatre limites détéctées.
    '''
    return cadrage_largeur(cadrage_hauteur(img))
