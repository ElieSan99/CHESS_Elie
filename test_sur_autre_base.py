from retranscrire_caracteres import retranscrire_caractere
import cv2 as cv
import os

dossier_images = 'destination'
for nom_image in os.listdir(dossier_images):
    chemin_image = os.path.join(dossier_images, nom_image)
    image = cv.imread(chemin_image, cv.IMREAD_GRAYSCALE)
    cv.imshow('caractere', image)
    cv.waitKey(0)
    print("caractère reconnu ", retranscrire_caractere(image))

cv.destroyAllWindows()