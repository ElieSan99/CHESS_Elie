import cv2 as cv
from cadrer_image import cadrage2


def affiche_ecran(image, titre):
    little = cv.resize(image,(300,700))
    cv.imshow(titre, little)
    cv.waitKey(0)


# Charger l'image
img = cv.imread("imagesFeuilles/IMG_0177.jpeg", cv.IMREAD_GRAYSCALE)
im0 = cv.resize(img,(600,1800))

cv.imshow('image originale', im0)
cv.waitKey(0)
    
# Appliquer la fonction de cadrage
img_cadree = cadrage2(img)

# Afficher les images
affiche_ecran(img_cadree, 'image cadree') 
