import cv2 as cv

def affiche_ecran(image, titre):
    cv.imshow(titre, image)
    cv.waitKey(0)

image_case = cv.imread("imagesCases/Cbd7.jpeg", cv.IMREAD_GRAYSCALE)
affiche_ecran(image_case, "case")
print("taille de l'image :", image_case.shape)

_,image_binarisee = cv.threshold(image_case, 85, 255, cv.THRESH_BINARY)
affiche_ecran(image_binarisee, "image binarisée")