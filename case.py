# juste pour des tests - pas dans le prog final

import cv2 as cv

def affiche_ecran(image, titre):
    cv.imshow(titre, image)
    cv.waitKey(0)

##################################################################################################

def isole_caracteres(case) :
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
    case --> image d'un coup avec plusieurs caracteres

    Retourne :
    liste_caracteres --> Liste des images des caractères détectés dans la case.

    '''
    desired_size = (64, 64)
    liste_caracteres = []
    #image_case = cv.convertScaleAbs(case) # ?????
    # _, thresh = cv.threshold(image_case, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU) image deja binarisee !!!
    contours,h = cv.findContours(255-case, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print('nb contours : ', len(contours))
    cpt = 0
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if (w>10) and (h>10):
            cpt += 1
            # Drawing a rectangle on copied image
            rect = cv.rectangle(case, (x, y), (x + w, y + h), (0, 255, 0), 2)
            affiche_ecran(rect, "car")
    print('nb rectangles : ', cpt)

    affiche_ecran(case, "toto")

    # contours = sorted(contours, key=lambda x: cv.boundingRect(x)[0])
    # print(contours)
    # rectangles_contours = []
    # for contour in contours :
    #     rectangles_contours.append(cv.boundingRect(contour))

        # rectangles_characteres = fusion_rectangles(rectangles_contours)
        # rectangles_characteres = supprimer_perturbations(rectangles_characteres, case.shape[1])
        # for rectangle in rectangles_characteres :
        #     x, y, w, h = rectangle
        #     contour_image = image_case[y : y + h, x : x + w]
        #     if h < desired_size[1] and w < desired_size[0]:
        #         pad_x = (desired_size[0] - w) // 2
        #         pad_y = (desired_size[1] - h) // 2
        #         contour_image = cv.copyMakeBorder(contour_image, pad_y, pad_y, pad_x, pad_x, cv.BORDER_CONSTANT, value=(255, 255, 255))
        #     contour_image = cv.resize(contour_image, desired_size)
        #     caracteres_case.append(contour_image)

        # if len( caracteres_case) == 0:
        #     liste_caracteres_par_image.append("vide")
        # else :
        #     liste_caracteres_par_image.append(caracteres_case)

    #return liste_caracteres_par_image

image_case = cv.imread("imagesCases/case43.jpg", cv.IMREAD_GRAYSCALE)
affiche_ecran(image_case, 'case 43')
isole_caracteres(image_case)