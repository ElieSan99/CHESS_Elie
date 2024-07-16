import cv2 as cv
from cadrer_image import cadrage
from detection_lignes import detection_lignes_feuille_cadree
from recuperation_cases import fourier, decoupage_case, create_dict
from retranscrire_caracteres import retranscrire_caractere

def affiche_ecran(image, titre):
    little = cv.resize(image,(300,700))
    cv.imshow(titre, little)
    cv.waitKey(0)

#Choisir l'image à traiter
#image_feuille = cv.imread("imagesFeuilles/Feuille_de_coup_23.jpg", cv.IMREAD_GRAYSCALE)
image_feuille = cv.imread("imagesFeuilles/Feuille176.jpeg", cv.IMREAD_GRAYSCALE) # 23.jpg
print("taille de l'image :", image_feuille.shape)
im0 = cv.resize(image_feuille,(600,1800))
cv.imshow('image originale ', im0)
cv.waitKey(0)
image_cadree = cadrage(image_feuille)
affiche_ecran(image_cadree, 'image cadree')
lignes_detectees = detection_lignes_feuille_cadree(image_cadree)
#print('lignes Hough : ', lignes_detectees)

image_fourier = fourier(image_cadree)
affiche_ecran(image_fourier, 'fourier')

liste_cases = decoupage_case(lignes_detectees, image_fourier)
# NB : y a du ménage à faire, certaines cases à oublier (comme la première = "BLANCS")

cv.imshow("une case", liste_cases[10])
cv.waitKey(0)
cv.imwrite('case10.jpg', liste_cases[10])

dict_case_caracteres = create_dict(liste_cases)

c = dict_case_caracteres[9]
print(c)
im = c[1]
cv.imshow('car1', im[0])
cv.waitKey(0)

cv.imshow('car2', im[1])
cv.waitKey(0)

cv.imshow('car3', im[2])
cv.waitKey(0)

cv.imshow('car4', im[3])
cv.waitKey(0)

print("caractère reconnu ", retranscrire_caractere(im[2]))

#On récupère ici les caractères détectés par le réseau de neurones, dans liste_coups.
liste_coups =[]
for i in range(10,11) :
    coups = f"{i}. "
    if dict_case_caracteres[i][0] != "vide" :
        for caractere in dict_case_caracteres[i][0] :
            coups += retranscrire_caractere(caractere)
    coups += " "
    if dict_case_caracteres[i][1] != "vide" :
        for caractere in dict_case_caracteres[i][1] :
            coups += retranscrire_caractere(caractere)
    liste_coups.append(coups)

print(liste_coups)
# closing all open windows 
cv.destroyAllWindows()