import cv2 as cv
from cadrer_image import cadrage, cadrage2
from detection_lignes import detection_lignes_feuille_cadree
from recuperation_cases import fourier, decoupage_case, create_dict, isolement_caracteres
from retranscrire_caracteres import retranscrire_caractere, retranscrire_caractere2, retranscrire_caractere3
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
image_feuille = cv.imread("imagesFeuilles/Feuille176.jpeg", cv.IMREAD_GRAYSCALE) # 23.jpg
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

liste_cases = decoupage_case(lignes_detectees, image_fourier)
# NB : y a du ménage à faire, certaines cases à oublier (comme la première = "BLANCS")

# Image binarisee
"""absInv = np.abs(image_cadree) # a enlever si bloc ci-dessus decommente
gris = absInv.astype(np.uint8) # conversion en niveaux de gris
_, image_binar = cv.threshold(gris, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
affiche_ecran(image_binar, "image binarise")"""


l = decoupage_case(lignes_detectees, image_cadree)
#ma_liste = isolement_caracteres(l)

"""
# Fonction pour afficher les caractères
def afficher_caracteres(liste_caracteres_par_image):
    for i, caracteres in enumerate(liste_caracteres_par_image):
        if caracteres == "vide":
            print(f"Case {i} : vide")
        else:
            print(f"Case {i} :")
            plt.figure(figsize=(10, 2))
            for j, caractere in enumerate(caracteres):
                plt.subplot(1, len(caracteres), j + 1)
                plt.imshow(caractere, cmap='gray')
                plt.axis('off')
            plt.show()
"""
#afficher_caracteres(ma_liste)
print(f"longueur liste 1: {len(l)}")
#print(f"longueur liste 1: {len(liste_cases)}")
#cv.imshow("une case", l[3])
#cv.waitKey(0)

#im_four = fourier(l[10])
#cv.imshow('case_fourier', im_four)
#cv.waitKey(0)

"""
absInv = np.abs(l[3]) # a enlever si bloc ci-dessus decommente
gris = absInv.astype(np.uint8) # conversion en niveaux de gris
_, res = cv.threshold(gris, 100, 255, cv.THRESH_BINARY)
cv.imshow('case_binaire', res)
cv.waitKey(0)
"""
#cv.imwrite('case10.jpg', liste_cases[10])

#dict_case_caracteres = create_dict(liste_cases)
dict_case_caracteres = create_dict(l)
#dict_case_caracteres

"""
c = dict_case_caracteres[2]

im = c[0]
cv.imshow('car1', im[0])
cv.waitKey(0)
cv.imwrite('car1.jpg', im[0])


print("caractère reconnu ", retranscrire_caractere(im[0]))
"""



#On récupère ici les caractères détectés par le réseau de neurones, dans liste_coups.
liste_coups =[]

"""for i in range(1,69) :
    #coups = f"{i}. "
    coup1 = ""
    if dict_case_caracteres[i][0] != "vide" :
        for caractere in dict_case_caracteres[i][0] :
            #cv.imshow('car', caractere)
            #cv.waitKey(0)
            #nom_fichier = f'caractere_degrade/car_{caractere_count}.jpg'
            #cv.imwrite(nom_fichier, caractere)
            car = retranscrire_caractere(caractere)
            coup1 += car
            #caractere_count += 1 
    #coups += " "

    # Tester coups
    


    if dict_case_caracteres[i][1] != "vide" :
        coup2 = ""
        for caractere in dict_case_caracteres[i][1] :
            #cv.imshow('car', caractere)
            #cv.waitKey(0)
            #nom_fichier = f'caractere_degrade/car_{caractere_count}.jpg'
            #cv.imwrite(nom_fichier, caractere)
            car = retranscrire_caractere(caractere)
            coup2 += car
            #caractere_count += 1
    #coup2 = tester_coups2(coup2)

    coups = f"{i}. " + coup1 + " " + coup2

    #print(coups)

    liste_coups.append(coups)


    with open('Liste_coups2.txt', 'w') as fichier:
    # 3. Écrire chaque caractère dans le fichier
        for coup in liste_coups:
            fichier.write(coup + '\n')
    

#print(liste_coups)
# closing all open windows 
cv.destroyAllWindows()"""

# test avec retranscrire_caractere2
jeu = chess.Board()
for i in range(1,69) :
   coup1 = retranscrire_caractere3(dict_case_caracteres[i][0], jeu)
   coup2 = retranscrire_caractere3(dict_case_caracteres[i][1], jeu)
   coups = f"{i}. " + coup1 + " " + coup2
   liste_coups.append(coups)

   with open('Liste_coups2_.txt', 'w') as fichier:
    # 3. Écrire chaque caractère dans le fichier
        for coup in liste_coups:
            fichier.write(coup + '\n')



# closing all open windows 
cv.destroyAllWindows()