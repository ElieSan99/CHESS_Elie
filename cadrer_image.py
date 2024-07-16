'''Import des differentes fonctions necessaires'''
from detection_lignes import detection_ligne_haut, detection_ligne_droite,\
        detection_ligne_gauche, detection_ligne_bas

import cv2 as cv
import numpy as np

def affiche_ecran(image, titre):
    little = cv.resize(image,(300,700))
    cv.imshow(titre, little)
    cv.waitKey(0)


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


def cadrage2(image):
    '''
    Fonction qui appelle les fonctions cadrage_vertical et cadrage_horizontal pour les appliquer.

    Paramètres :
    img (array) : Une liste de listes, ayant pour valeurs les pixels aux niveaux de gris
    de la photo initale.

    Retourne :
    img_cadree (array) : Une partie du paramètre en entrée, contenant uniquement les données
    comprises entre les quatre limites détectées.
    '''
   
    
    longueur_image = image.shape[0]
    #print(f"image largeur : {largeur_image}")
    print(f"image longueur : {longueur_image}")

    #detection ligne haut
    ligne_haut = detection_ligne_haut2(image)

    # Déterminer les dimensions de l'image originale
    height, width = image.shape[:2]

    # Rogner l'image
    cropped_image = image[ligne_haut:height, 0:width]
    im0 = cv.resize(cropped_image,(600,1800))
    cv.imshow('image originale', im0)
    cv.waitKey(0)
    
    largeur_image = cropped_image.shape[1]

    image_contour = cv.Canny(cropped_image, 20, 200, None, 3)
    #affiche_ecran(image_contour, 'image contour')
    kernel = np.ones((3,3), np.uint8)
    image_contour = cv.dilate(image_contour, kernel, iterations=1)
    #affiche_ecran(image_contour, 'image contour apres dilatation')

    lignes_verticales = cv.HoughLinesP(image_contour, 1, np.pi / 180, 350, None, 400, 10)
    """
    if lignes_verticales is not None:
        for ligne in lignes_verticales:
            for x1, y1, x2, y2 in ligne:
                cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 5) 
    """
    
    #print(f"lignes verticales {lignes_verticales}")

    lignes_horizontales = cv.HoughLinesP(image_contour, 1, np.pi / 180,
                                         350, None, largeur_image*0.06, 10)
    
    #print(f"lignes horizontales {lignes_horizontales}")
    """
    if lignes_horizontales is not None:
        for ligne in lignes_horizontales:
            for x1, y1, x2, y2 in ligne:
                cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 5) 
    """
    # Afficher l'image avec les lignes détectées
    #affiche_ecran(image,'Image avec lignes')

    liste_lignes = [[], []]
    seuil = 60
    for points in lignes_verticales :
        x_1, x_2 = points[0][0], points[0][2]
        if abs(x_1 - x_2) < seuil:
            liste_lignes[0].append(int((x_1 + x_2)/2))

    for points in lignes_horizontales :
        y_1, y_2 = points[0][1], points[0][3]
        if abs(y_1 - y_2) < seuil:
            liste_lignes[1].append(int((y_1 + y_2)/2))
    
    print("Lignes verticales détectées:", liste_lignes[0])
    print("Lignes horizontales détectées:", liste_lignes[1])

    liste_lignes_triee = supression_lignes_successives2(liste_lignes)

    print("Lignes verticales après suppression:", liste_lignes_triee[0])
    print("Lignes horizontales après suppression:", liste_lignes_triee[1])

    # Cadrer l'image en utilisant les lignes détectées
    v = liste_lignes_triee[0]
    print(f"nb lignes verticales : {len(v)}")
    h = liste_lignes_triee[1]
    print(f"nb lignes horizontales : {len(h)}")

   
    while len(h)>42:
        h.pop()

    min_x, max_x = min(v), max(v)
    min_y, max_y = min(h), max(h)
    #min_x +=30
    #max_x -=65
    #max_y -=140
    min_y -=10
    image_cadree = cropped_image[min_y:max_y, min_x:max_x]
    print(f"nb lignes horizontales apres : {len(h)}")
    return image_cadree


def detection_ligne_haut2(image) :
    '''
    Cette fonction permet de trouver la limite haute à laquelle il faut couper
    la feuille.

    La limite haute est la ligne sur laquelle se trouvent les deux cases
    ayant pour inscription "BLANC" et "NOIR". 

    Pour cela, on binarise, puis dilate l'image, afin d'obtenir uniquement les
    coordonnees des cases precedentes. 

    Parametre :
    image(array) --> Liste comprenant les valeurs de l'image initiale.

    Retourne :
    hauteur_limite(int) --> Indice de la première ligne détectée.
    '''
    _,image_binarisee = cv.threshold(image, 60, 255, cv.THRESH_BINARY)
    #_,image_binarisee = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,9,0)
    affiche_ecran(image_binarisee,'image binarisee ')
    #cv.waitKey(0)
    kernel = np.ones((10,10), np.uint8) # kernel (10,10) initialement
    # on ne veut garder que des blocs noirs correspondant aux mots "noirs"
    # qui serviront de repère (fin de l'en-tete)
    image_erodee = cv.dilate(image_binarisee, kernel, iterations = 3) # nb iterations = 3 initialement
    #affiche_ecran(image_erodee, 'image erodee')
    
    _, nbcol = image.shape

    cptmax = 0
    imax = 0
    for i in range(1000):
        cpt = 0
        for j in range(nbcol):
            if image_erodee[i][j] == 0:
                cpt += 1
        if cpt>cptmax:
            imax = i
            cptmax = cpt 

    print("ligne haut = ", imax)

    return imax



def supression_lignes_successives2(liste_lignes) :
    '''
    Cette fonction a pour but de remplacer les indices des lignes successifs
    en un seul indice moyen.

    Parametre :
    liste_lignes(list) --> Liste comprenant les indices des lignes détectées.

    Retourne :
    - liste_lignes_triee_v(array) --> Liste triée comme annoncé précédemment,
    comprenant les indices des lignes verticales détectées.
    - liste_lignes_triee_h(array) --> Liste triée comme annoncé précédemment,
    comprenant les indices des lignes horizontales détectées.
    '''
    liste_lignes[0].sort(), liste_lignes[1].sort()
    #print("Lignes verticales triées:", liste_lignes[0])
    #print("Lignes horizontales triées:", liste_lignes[1])
    liste_lignes_triee_v = []
    liste_lignes_triee_h = []
    prec_ind = [liste_lignes[0][0]]

    seuil1 = 110

    seuil2 = 15

    for index in liste_lignes[0][1:] :
        if index - prec_ind[-1] > seuil1 :
            liste_lignes_triee_v.append(int(np.mean(prec_ind)))
            prec_ind = [index]
        else :
            prec_ind.append(index)
    liste_lignes_triee_v.append(int(np.mean(prec_ind)))

    prec_ind = [liste_lignes[1][0]]

    for index in liste_lignes[1][1:] :
        if index - prec_ind[-1] > seuil2 :
            liste_lignes_triee_h.append(int(np.mean(prec_ind)))
            prec_ind = [index]
        else :
            prec_ind.append(index)
    liste_lignes_triee_h.append(int(np.mean(prec_ind)))

    return liste_lignes_triee_v, liste_lignes_triee_h
