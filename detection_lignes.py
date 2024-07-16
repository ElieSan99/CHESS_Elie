'''OpenCV pour traiter les images'''
import numpy as np
import cv2 as cv

def affiche_ecran(image, titre):
    little = cv.resize(image,(300,700))
    cv.imshow(titre, little)
    cv.waitKey(0)

def detection_ligne_haut (image) :
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
    _,image_binarisee = cv.threshold(image, 85, 255, cv.THRESH_BINARY)
    #_,image_binarisee = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,9,0)
    #affiche_ecran(image_binarisee,'image binarisee ')
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

def detection_ligne_bas(image) :
    '''
    Cette fonction permet de trouver la limite basse à laquelle il faut
    couper la feuille.

    La limite basse est la ligne à laquelle se termine la feuille d'échec initiale. 

    Pour cela, on binarise l'image, et on considère que si plus de 100 pixels qui se
    suivent sont noirs, alors on a une ligne. Ici, on parcours l'image en partant
    de la fin, et dès que l'on détecte une ligne, on est à la limite basse. 

    Parametre :
    image(array) --> Liste comprenant les valeurs de l'image initiale.

    Retourne :
    limite_bas(int) --> Indice de la premiere ligne detectee en partant de la fin.
    '''
    imb = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,55,1)
    affiche_ecran(imb, 'image binarisee 2')
    limite_bas = -1
    hauteur_image = image.shape[0]
    print(f"hauteur de l'image {hauteur_image}")
    for i in range (hauteur_image-1,0,-1):
        noir = 0
        for j in imb[i]:
            if j == 0:
                noir += 1
        if noir > 1500 :
            limite_bas = i
            break
    print(f"Limite bas {limite_bas}")
    return limite_bas

def detection_ligne_gauche(image) :
    '''
    Cette fonction permet de trouver la limite gauche à laquelle il faut
    couper la feuille.

    La limite gauche est l'indice des colonne pour lequel se termine la
    feuille d'échec initiale, du côté gauche. 

    Pour cela, on binarise l'image, et on considère que si plus de 100 pixels qui se
    suivent sont noirs, alors on a une ligne. Ici, on transpose l'image pour faciliter
    le parcours des colonnes. Dès qu'une ligne est détectée, on est à la limite gauche. 

    Parametre :
    image(array) --> Liste comprenant les valeurs de l'image initiale.

    Retourne :
    limite_bas(int) --> Indice de la premiere ligne detectee, en partant de la fin pour
    être du côté gauche (cela est dû à la transposition).
    '''
    image_bin = cv.threshold(image,80,255, cv.THRESH_BINARY)[1]
    largeur_image = image.shape[1]
    for i in range (largeur_image-1,0,-1):
        noir = 0
        for j in image_bin.T[i]:
            if j == 0:
                noir += 1
        if noir > 100:
            limite_gauche = i
            break
    print(f"Limite gauche {limite_gauche}")
    return limite_gauche

def detection_ligne_droite(image) :
    '''
    Cette fonction permet de trouver la limite droite à laquelle il faut
    couper la feuille.

    La limite droite est l'indice des colonne pour lequel se termine la
    feuille d'échec initiale, du côté droit. 

    Pour cela, on binarise l'image, et on considère que si plus de 100 pixels qui se
    suivent sont noirs, alors on a une ligne. Ici, on transpose l'image pour faciliter
    le parcours des colonnes. Dès qu'une ligne est détectée, on est à la limite droite. 

    Parametre :
    image(array) --> Liste comprenant les valeurs de l'image initiale.

    Retourne :
    limite_bas(int) --> Indice de la premiere ligne detectee en partant de la fin.
    '''
    _, image_bin = cv.threshold(image,80,255, cv.THRESH_BINARY)
    largeur_image = image.shape[1]
    for i in range (largeur_image):
        noir = 0
        for j in image_bin.T[i]:
            if j == 0:
                noir += 1
        if noir > 100:
            limite_droite = i
            break
    print(f"Limite droite {limite_droite}")
    return limite_droite

def detection_lignes_feuille_cadree(image) :
    '''
    Cette fonction détecte les lignes verticales et horizontales sur la feuille. 
    On utilise pour cela la fonction HoughlinesP de Opencv (qui detecte les lignes).

    Parametre :
    image(array) --> Liste comprenant les valeurs de l'image initiale.

    Retourne :
    liste_lignes_triee --> Liste comprenant les indices des lignes détectées.
    Elle est composée de deux listes, la première contenant les lignes verticales,
    et la deuxième les horizontales.
    '''

    largeur_image = image.shape[1]

    image_contour = cv.Canny(image, 20, 200, None, 3)
    kernel = np.ones((3,3), np.uint8)
    image_contour = cv.dilate(image_contour, kernel, iterations=1)

    lignes_verticales = cv.HoughLinesP(image_contour, 1, np.pi / 180, 350, None, 500, 4)

    #print(f"lignes verticales {lignes_verticales}")

    lignes_horizontales = cv.HoughLinesP(image_contour, 1, np.pi / 180,
                                         350, None, largeur_image*0.1, 5)
    
    #print(f"lignes horizontales {lignes_horizontales}")

    liste_lignes = [[], []]

    for points in lignes_verticales :
        x_1, x_2 = points[0][0], points[0][2]
        if abs(x_1 - x_2) < 60:
            liste_lignes[0].append(int((x_1 + x_2)/2))

    for points in lignes_horizontales :
        y_1, y_2 = points[0][1], points[0][3]
        if abs(y_1 - y_2) < 60:
            liste_lignes[1].append(int((y_1 + y_2)/2))
    
    print("Lignes verticales détectées:", liste_lignes[0])
    print("Lignes horizontales détectées:", liste_lignes[1])

    liste_lignes_triee = supression_lignes_successives(liste_lignes)

    print("Lignes verticales après suppression:", liste_lignes_triee[0])
    print("Lignes horizontales après suppression:", liste_lignes_triee[1])

    return liste_lignes_triee

def supression_lignes_successives(liste_lignes) :
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
    liste_lignes_triee_v = []
    liste_lignes_triee_h = []
    prec_ind = [liste_lignes[0][0]]

    for index in liste_lignes[0][1:] :
        if index - prec_ind[-1] > 15 :
            liste_lignes_triee_v.append(int(np.mean(prec_ind)))
            prec_ind = [index]
        else :
            prec_ind.append(index)
    liste_lignes_triee_v.append(int(np.mean(prec_ind)))

    prec_ind = [liste_lignes[1][0]]

    for index in liste_lignes[1][1:] :
        if index - prec_ind[-1] > 15 :
            liste_lignes_triee_h.append(int(np.mean(prec_ind)))
            prec_ind = [index]
        else :
            prec_ind.append(index)
    liste_lignes_triee_h.append(int(np.mean(prec_ind)))

    return liste_lignes_triee_v, liste_lignes_triee_h
