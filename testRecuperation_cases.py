import unittest
import numpy as np
from recuperation_cases import decoupage_case, create_dict, fourier, isolement_caracteres
from detection_lignes import detection_lignes_feuille_cadree
from cadrer_image import cadrage, cadrage2
import cv2 as cv

class TestMonModule(unittest.TestCase):
    
    def setUp(self):
        self.image = cv.imread("imagesFeuilles/Feuille176.jpeg", cv.IMREAD_GRAYSCALE)
    
    """
    def test_decoupage_case(self):
       
        
        print("taille de l'image :",  self.image.shape)
        #cv.imshow('image originale ', im0)
        #cv.waitKey(0)
        image_cadree = cadrage( self.image)
        detect = detection_lignes_feuille_cadree(image_cadree)
        # Créer une image de test
        image_fourier_test = fourier(image_cadree)
        # Vérifier si la fonction decoupage_case ne renvoie pas d'erreur
        self.assertIsNotNone(decoupage_case(detect, image_fourier_test))
    """

    def test_create_dict(self):
        image_cadree = cadrage( self.image)
        detect = detection_lignes_feuille_cadree(image_cadree)
        # Créer une image de test
        image_fourier_test = fourier(image_cadree)
        # Créer une liste de cases de test
        liste_cases_test = decoupage_case(detect, image_fourier_test)
        # Vérifier si la fonction create_dict2 ne renvoie pas d'erreur
        self.assertIsNotNone(create_dict(liste_cases_test))
    
"""
    def test_isolement_caracteres(self):
        image_cadree = cadrage( self.image)
        detect = detection_lignes_feuille_cadree(image_cadree)
        # Créer une image de test
        image_fourier_test = fourier(image_cadree)
        # Créer une liste de cases de test
        liste_cases_test = decoupage_case(detect, image_fourier_test)
        self.assertIsNotNone(isolement_caracteres(liste_cases_test))
"""
        

if __name__ == '__main__':
    unittest.main()
