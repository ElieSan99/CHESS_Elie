"""A conserver"""
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

def fourier(image) :
    '''
    Cette fonction a pour but d'appliquer une transformation de Fourier sur
    l'image cadrée, afin de supprimer les lignes de la feuille (périodiques).

    La fonction utilise un paramètre qui doit être adapté à chaque image,
    ici appelé factor (ligne 29)

    Paramètre :
    image(array) --> Liste comprenant les valeurs de l'image initiale.

    Retourne :
    image(array) --> Liste comprenant les valeurs de l'image après traitement.
    Pour la rendre plus visible, l'image en sortie est dilatée.
    '''

    # Calculer la transformee de Fourier 2D de l'image filtrée
    image_fourier = np.fft.fft2(image)
    # Pour mettre les coefficients basse fréquence au centre
    fshift = np.fft.fftshift(image_fourier)
    # Supprimer les fréquences de la grille
    lignes, colonnes = image.shape
    ligne_milieu, colonne_milieu = lignes//2, colonnes//2
    #mask = np.zeros((lignes, colonnes), np.uint8)
    #factor = 1/2.9
    # factor = 1.0/3.0
    # mask[int(ligne_milieu*(1 - factor)) : int(ligne_milieu*(1 + factor)),
    #      int(colonne_milieu*(1 - factor)) : int(colonne_milieu*(1 + factor))] = 1
    # f_transform = fshift * mask # NB : produit membre a membre (pas le produit matriciel)

    marge = 5
    fshift[:, colonne_milieu - marge:colonne_milieu + marge] = 0

    # Calculer la transformée de Fourier inverse de l'image filtrée
    f_ishift = np.fft.ifftshift(fshift)
    image_back = np.fft.ifft2(f_ishift)
    # seuil = 100
    # _, bin = cv.threshold(np.abs(image_back), seuil, 255, cv.THRESH_BINARY)
    # kernel = np.ones((3, 3), np.uint8)
    # erodee = cv.erode(resultat, kernel, iterations=1)
    absInv = np.abs(image_back) # a enlever si bloc ci-dessus decommente
    gris = absInv.astype(np.uint8) # conversion en niveaux de gris
    _, res = cv.threshold(gris, 55, 255, cv.THRESH_BINARY_INV)
    
    return res


def decoupage_case(detect, image_fourier) :
    '''
    Repartit l'image de Fourier en cases individuelles en utilisant les coordonnées fournies
    par la détection des lignes.

    Paramètres :
    - detect (tuple) --> Cet élément contient les listes des coordonnées des colonnes
    et des lignes, respectivement dans cet ordre.
    - image_fourier (array) --> Liste ayant les valeurs de l'image après transformée de Fourier.

    Returns:
    liste_case (array) --> Liste des cases découpées, à partir de l'image de Fourier.
    '''
    liste_cases = []
    colonnes = detect[0]
    #print(f"colonnes {colonnes}")
    #print(f"nb col {len(colonnes)}")
    #x = range(len(colonnes))  # x sera [0, 1, 2, 3, 4, 5]
    #plt.plot(x, colonnes, marker='o', linestyle='-', color='b', label='colonnes')
    #plt.show()

    lignes = detect[1]
    del lignes[:1]
    lignes.pop()
    #print(f"lignes {lignes}")
    #x = range(len(lignes))  # x sera [0, 1, 2, 3, 4, 5]
    #plt.plot(x, lignes, marker='o', linestyle='-', color='b', label='Lines')
    #plt.show()
    #print(f"nb ligne {len(lignes)}")
    colonne_prec = colonnes[0] 
    for colonne in colonnes[1:] :
        ligne_prec = lignes[0]
        for ligne in lignes[1:] :
            if colonne_prec == colonnes[0]:
                c1 = colonne_prec - 20
            else:
                c1 = colonne_prec + 10
            c2 = colonne - 30
            l1 = ligne_prec + 5
            l2 = ligne
            liste_cases.append(image_fourier[l1 : l2, c1 : c2])
            ligne_prec = ligne
        colonne_prec = colonne
    #On procède à la suppression des nombres des lignes pour les cases les contenant.
    liste_cases[:40] = suppresion_index(liste_cases[:40])
    liste_cases[80:120] = suppresion_index(liste_cases[80:120])

    """
    for case in liste_cases[120:125]:
        cv.imshow('decoupage case', case)
        cv.waitKey(0)
    """

    return liste_cases

def suppresion_index(liste_cases_avec_index) :
    '''
    Supprime la partie de l'image ou le nombre des lignes apparait. Cette fonction n'est appellée
    que pour les cases contenant les coups du joueur blanc.

    Paramètre :
    liste_cases_avec_index (list) --> Liste des images des cases,
    comprenant les nombres des lignes.
    
    Retourne :
    liste_cases (list) --> Liste des images des cases,
    ne comprenant plus les nombres des lignes.
    '''
    liste_cases_sans_index = []
    for case in liste_cases_avec_index :
        liste_cases_sans_index.append(case[:, 75:])
    return liste_cases_sans_index

def isolement_caracteres(liste_cases) :
    '''
    A l'aide de la bibliothèque OpenCV, on détecte les composants noirs sur chaque case.

    Pour cela, on utilise les fonctions cv.findContours pour obtenir dans un premier temps
    les coordonnées des coins du rectangle encadrant le composant noir

    Ensuite on utilise cv.boundingRect, qui est le rectangle correspondant.
    Un rectangle est une liste, de la forme [x, y, w, h], avec les informations suivantes :
    - x : coordonnée de départ suivant la largeur.
    - y : coordonnée de départ suivant la hauteur.
    - w : (width) la longueur du rectangle.
    - h : (height) la hauteur du rectangle.

    Paramètre :
    liste_cases (list) --> Liste des images des cases.

    Retourne :
    liste_caracteres_par_image (list) --> Liste des images des caractères détectés par case.

    '''
    desired_size = (128, 128) # le RN prend des images 128x128 et il était écrit 64x64 !!
    liste_caracteres_par_image = []
    for case in liste_cases :
        caracteres_case = []
        contours, _ = cv.findContours(255 - case, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv.boundingRect(x)[0]) # tri des caractères selon ordre lexicographique !
        liste_rectangles = []
        for contour in contours :
            x, y, w, h = cv.boundingRect(contour)
            if (w>10) and (h>10):
                liste_rectangles.append([x, y, w, h])

        for rectangle in liste_rectangles :
            x, y, w, h = rectangle
            contour_image1 = case[y : y + h, x : x + w]
            contour_image2 = cv.resize(contour_image1, desired_size)
            caracteres_case.append(contour_image2) 

        if len( caracteres_case) == 0:
            liste_caracteres_par_image.append("vide")
        else :
            liste_caracteres_par_image.append(caracteres_case)

    return liste_caracteres_par_image


def create_dict(liste_cases) :
    '''
    Cette fonction permet de créer le dictionnaire comprenant les différentes images
    des caractères, par coup et par joueur.
    Il peut être appelé de la façon suivante : dict_caractere[i][j][k]
    Avec :
    - i le nombre du coup (entre 1 et 80 inclus)
    - j si le joueur est blanc ou noir (0 si blanc, 1 sinon)
    - k le numéro du caractère voulu

    Paramètre :
    liste_cases (list) --> Liste des cases de la feuille.

    Retourne :
    dict_caractere (dict) --> Dictionnaire comprenant les images des différents caractères
    des coups joués.
    '''
    liste_caracteres_par_image = isolement_caracteres(liste_cases)
    dict_caractere = {i : [] for i in range(1,81)}
    for num in range (1, 41) :

        dict_caractere[num] = [
            liste_caracteres_par_image[num-1],
            liste_caracteres_par_image[39 + num]
            ]

        dict_caractere[40 + num] = [
            liste_caracteres_par_image[79 + num],
            liste_caracteres_par_image[119 + num]
            ]

    return dict_caractere


def create_dict2(liste_cases):
    '''
    Cette fonction permet de créer le dictionnaire comprenant les différentes images
    des caractères, par coup et par joueur.
    Il peut être appelé de la façon suivante : dict_caractere[i][j][k]
    Avec :
    - i le nombre du coup (entre 1 et 80 inclus)
    - j si le joueur est blanc ou noir (0 si blanc, 1 sinon)
    - k le numéro du caractère voulu

    Paramètre :
    liste_cases (list) --> Liste des cases de la feuille.

    Retourne :
    dict_caractere (dict) --> Dictionnaire comprenant les images des différents caractères
    des coups joués.
    '''
    liste_caracteres_par_image = isolement_caracteres(liste_cases)
    dict_caractere = {i: [[], []] for i in range(1, 81)}

    for num in range(1, 41):
        # Premier set de coups (1 à 40)
        dict_caractere[num][0] = liste_caracteres_par_image[(num - 1) * 2]
        dict_caractere[num][1] = liste_caracteres_par_image[(num - 1) * 2 + 1]

        # Deuxième set de coups (41 à 80)
        dict_caractere[40 + num][0] = liste_caracteres_par_image[(num + 39) * 2]
        dict_caractere[40 + num][1] = liste_caracteres_par_image[(num + 39) * 2 + 1]

    return dict_caractere
