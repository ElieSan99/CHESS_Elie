import unittest
import cv2
from detection_lignes import *

class TestDetectionLignes(unittest.TestCase):

    def setUp(self):
        self.image = cv2.imread('image_cadree.jpg', cv2.IMREAD_GRAYSCALE)

    def test_detection_ligne_haut(self):
        limite_haut = detection_ligne_haut(self.image)
        self.assertIsNotNone(limite_haut)
        self.assertIsInstance(limite_haut, int)
        self.assertGreaterEqual(limite_haut, 0)

    def test_detection_ligne_bas(self):
        limite_bas = detection_ligne_bas(self.image)
        self.assertIsNotNone(limite_bas)
        self.assertIsInstance(limite_bas, int)
        

    def test_detection_ligne_gauche(self):
        limite_gauche = detection_ligne_gauche(self.image)
        self.assertIsNotNone(limite_gauche)
        self.assertIsInstance(limite_gauche, int)
        self.assertGreaterEqual(limite_gauche, 0)

    def test_detection_ligne_droite(self):
        limite_droite = detection_ligne_droite(self.image)
        self.assertIsNotNone(limite_droite)
        self.assertIsInstance(limite_droite, int)
        self.assertGreaterEqual(limite_droite, 0)

    def test_detection_lignes_feuille_cadree(self):
        liste_lignes_verticales, liste_lignes_horizontales = detection_lignes_feuille_cadree(self.image)
        self.assertIsNotNone(liste_lignes_verticales)
        self.assertIsNotNone(liste_lignes_horizontales)
        self.assertIsInstance(liste_lignes_verticales, list)
        self.assertIsInstance(liste_lignes_horizontales, list)
        self.assertGreater(len(liste_lignes_verticales), 0)
        self.assertGreater(len(liste_lignes_horizontales), 0)

if __name__ == '__main__':
    unittest.main()
