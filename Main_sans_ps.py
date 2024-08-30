import cv2 as cv
from cadrer_image import cadrage, cadrage2
from detection_lignes import detection_lignes_feuille_cadree
from recuperation_cases import fourier, decoupage_case, create_dict, isolement_caracteres
from retranscrire_caracteres import retranscrire_caractere, post_traitement, proportion_coups_traite_correct, InteractionUser
import numpy as np
import matplotlib.pyplot as plt
from test_chess import transformation_coups_en_liste, transformation_coups_en_txt, tester_coups
import chess
import chess.pgn



def affiche_ecran(image, titre):
    little = cv.resize(image,(300,700))
    cv.imshow(titre, little)
    cv.waitKey(0)

#Choisir l'image à traiter
##########################
image_feuille = cv.imread("imagesFeuilles/IMG20231107104606.jpg", cv.IMREAD_GRAYSCALE) # 23.jpg
print("taille de l'image :", image_feuille.shape)
im0 = cv.resize(image_feuille,(600,1800))
# Afficher l'image originale
#cv.imshow('image originale ', im0)
#cv.waitKey(0)

# Cadrer l'image
image_cadree = cadrage2(image_feuille)
#affiche_ecran(image_cadree, 'image cadree')

# Appliquer la transformée de Fourier
lignes_detectees = detection_lignes_feuille_cadree(image_cadree)
print('lignes Hough : ', lignes_detectees)
cv.imwrite('image_cadree.jpg', image_cadree)
image_fourier = fourier(image_cadree)
#affiche_ecran(image_fourier, 'fourier')

# decoupage des cases
liste_cases = decoupage_case(lignes_detectees, image_cadree)

# on recupère dans un dictionnaire la liste des images des caractères des joueurs
dict_case_caracteres = create_dict(liste_cases)


#On récupère ici les caractères détectés par le réseau de neurones, dans liste_coups.
liste_coups =[]

for i in range(1,69) :
    coup1 = ""
    coup2 = ""
    if dict_case_caracteres[i][0] != "vide" :
        #n = 0
        for caractere in dict_case_caracteres[i][0] :
            
            car = retranscrire_caractere(caractere)
            coup1 += car


    if dict_case_caracteres[i][1] != "vide" :
        for caractere in dict_case_caracteres[i][1] :
            car = retranscrire_caractere(caractere)
            coup2 += car

    coups = f"{i}. " + coup1 + " " + coup2

    liste_coups.append(coups)


    with open('Liste_coups2.txt', 'w') as fichier:
    # 3. Écrire chaque caractère dans le fichier
        for coup in liste_coups:
            fichier.write(coup + '\n')
    

#print(liste_coups)
# closing all open windows 
cv.destroyAllWindows()

