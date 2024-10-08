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
#image_fourier = fourier(image_cadree)
#affiche_ecran(image_fourier, 'fourier')

# decoupage des cases
liste_cases = decoupage_case(lignes_detectees, image_cadree)

# on recupère dans un dictionnaire la liste des images des caractères des joueurs
dict_case_caracteres = create_dict(liste_cases)



#On récupère ici les caractères détectés par le réseau de neurones, dans liste_coups.
liste_coups =[]

total_coups = 0  # Compteur pour le nombre total de coups
intervention = InteractionUser()
jeu = chess.Board()
for i in range(1, 42):
    print(f"Ligne {i} : coup 1")
    coup1 = post_traitement(dict_case_caracteres[i][0], jeu, intervention)
    #coup1 = postraitement_sans_user(dict_case_caracteres[i][0], jeu)
    
    if coup1 is None:
        print(f"Fin du programme : coup1 vide à la ligne {i}")
        break

    print(f"Ligne {i} : coup 2")
    coup2 = post_traitement(dict_case_caracteres[i][1], jeu, intervention)
    #coup2 = postraitement_sans_user(dict_case_caracteres[i][1], jeu)
    
    if coup2 is None:
        coup2 = ""

    coups = f"{i}. " + coup1 + " " + coup2
    liste_coups.append(coups)
    total_coups += 2  # Incrémenter le nombre total de coups

    with open('Liste_coups2_.txt', 'w') as fichier:
    # 3. Écrire chaque caractère dans le fichier
        for coup in liste_coups:
            fichier.write(coup + '\n')

interventions_utilisateur = intervention.get_interventions()
print(f"nb interventions : {interventions_utilisateur}")
# Calcul de la proportion d'intervention de l'utilisateur
if total_coups > 0:
    proportion_intervention = (interventions_utilisateur / total_coups)*100
    print(f"Proportion d'intervention de l'utilisateur : {proportion_intervention:.2f} %")
else:
    print("Aucun coup joué.")



#print(f"{proportion_coups_traite_correct()} %")

# closing all open windows 
cv.destroyAllWindows()

