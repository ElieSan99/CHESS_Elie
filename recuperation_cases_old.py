"""A conserver"""
import numpy as np
import cv2 as cv
from PIL import Image

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
    _, res = cv.threshold(gris, 60, 255, cv.THRESH_BINARY_INV)
    
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
    lignes = detect[1]
    colonne_prec = colonnes[0]
    for colonne in colonnes[1:] :
        ligne_prec = lignes[0]
        for ligne in lignes[1:] :
            liste_cases.append(image_fourier[ligne_prec : ligne, colonne_prec : colonne])
            ligne_prec = ligne
        colonne_prec = colonne
    #On procède à la suppression des nombres des lignes pour les cases les contenant.
    liste_cases[:40] = suppresion_index(liste_cases[:40])
    liste_cases[80:120] = suppresion_index(liste_cases[80:120])
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
        liste_cases_sans_index.append(case[:, 40:])
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
    desired_size = (64, 64)
    liste_caracteres_par_image = []
    for case in liste_cases :
        caracteres_case = []
        image_case = cv.convertScaleAbs(case)
        thresh = cv.threshold(image_case, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv.boundingRect(x)[0])
        rectangles_contours = []
        for contour in contours :
            rectangles_contours.append(cv.boundingRect(contour))

        rectangles_characteres = fusion_rectangles(rectangles_contours)
        rectangles_characteres = supprimer_perturbations(rectangles_characteres, case.shape[1])
        for rectangle in rectangles_characteres :
            x, y, w, h = rectangle
            contour_image = image_case[y : y + h, x : x + w]
            if h < desired_size[1] and w < desired_size[0]:
                pad_x = (desired_size[0] - w) // 2
                pad_y = (desired_size[1] - h) // 2
                contour_image = cv.copyMakeBorder(contour_image, pad_y, pad_y, pad_x, pad_x, cv.BORDER_CONSTANT, value=(255, 255, 255))
            contour_image = cv.resize(contour_image, desired_size)
            caracteres_case.append(contour_image)

        if len( caracteres_case) == 0:
            liste_caracteres_par_image.append("vide")
        else :
            liste_caracteres_par_image.append(caracteres_case)

    return liste_caracteres_par_image

def fusion_rectangles(liste_rectangles) :
    '''
    Cette fonction permet de fusionner les composants noirs détectés.
    Le but de ce processus est de fusionner les parties des lettres qui sont le plus dégradées.

    Paramètre :
    liste_rectangles (list) --> Liste des rectangles des composants noirs détectés.
    
    Retourne :
    liste_rectangles_fusionnes (list) --> Liste des rectangles fusionnés ou non.
    '''
    if len(liste_rectangles) == 0 :
        return []
    liste_rectangles_fusionnes = []
    prec_rectangle = liste_rectangles[0]
    for rectangle in liste_rectangles[1:] :

        if (
            (rectangle[0] - (prec_rectangle[0] + prec_rectangle[2]) < -5) and
             ((rectangle[1] < (prec_rectangle[1] + prec_rectangle[3]) and
               rectangle[1] > prec_rectangle[1]) or
              (prec_rectangle[1] < (rectangle[1] + rectangle[3]) and
               prec_rectangle[1] > rectangle[1]))
             ):

            new_rectangle = [
                prec_rectangle[0],
                min(prec_rectangle[1], rectangle[1]),
                max(prec_rectangle[0] + prec_rectangle[2], rectangle[0] + rectangle[2]) \
                    - prec_rectangle[0],
                max(prec_rectangle[1] + prec_rectangle[3], rectangle[1] + rectangle[3]) \
                    - min(prec_rectangle[1], rectangle[1])
                ]
            liste_rectangles.remove(rectangle)
            prec_rectangle = new_rectangle
        else :
            liste_rectangles_fusionnes.append(prec_rectangle)
            prec_rectangle = rectangle
    liste_rectangles_fusionnes.append(prec_rectangle)

    return liste_rectangles_fusionnes

def supprimer_perturbations(liste_rectangles, largeur_image) :
    '''
    Cette fonction permet de supprimer certains les composants noirs détectés.
    Le but de ce processus est de supprimer les parties détectées n'étant pas des caractères.

    Paramètre :
    liste_rectangles (list) --> Liste des rectangles des composants noirs détectés.
    
    Retourne :
    liste_rectangles_triee (list) --> Liste des rectangles gardés.
    '''
    liste_rectangles_triee = []
    for rectangle in liste_rectangles :
        if (rectangle[2] > 10 or rectangle[3] > 10) and\
            rectangle[1] + rectangle[3]> int(largeur_image*0.2) :
            liste_rectangles_triee.append(rectangle)

    return liste_rectangles_triee



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
