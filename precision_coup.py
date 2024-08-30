"""def precision_caractere(liste_coups_generes_txt, liste_coups_corrects_txt):
    # Lire la liste de coups générés
    with open(liste_coups_generes_txt, 'r') as fichier:
        liste_coups_predits = fichier.readlines()

    # Lire la liste de coups attendus
    with open(liste_coups_corrects_txt, 'r') as fichier:
        liste_coups_reels = fichier.readlines()

    # Nettoyer les listes (supprimer les espaces blancs en début/fin de ligne)
    liste_coups_predits = [coup.strip() for coup in liste_coups_predits]
    liste_coups_reels = [coup.strip() for coup in liste_coups_reels]

    # Comparer les listes et compter les coups corrects
    assert len(liste_coups_predits) == len(liste_coups_reels)

    # Initialiser les compteurs
    caracteres_corrects = 0
    total_caracteres = 0

    # Comparer les coups prédits aux coups réels
    for coups_reel, coups_pred in zip(liste_coups_reels, liste_coups_predits):
        for caractere_reel, caractere_pred in zip(coups_reel, coups_pred):
            if caractere_reel == caractere_pred:
                caracteres_corrects += 1
            total_caracteres += 1

    # Calculer le taux de bonnes prédictions
    taux_bonnes_predictions = (caracteres_corrects / total_caracteres) * 100

    return taux_bonnes_predictions


taux = precision_caractere('Feuille176_post_traitement.txt', 'Feuille_176_verite_terrain.txt')
print(f"Taux de bonnes prédictions : {taux:.0f}%")
"""

def precision_coup(liste_coups_generes_txt, liste_coups_corrects_txt):
    # Lire la liste de coups générés
    with open(liste_coups_generes_txt, 'r') as fichier:
        liste_coups_predits = fichier.readlines()

    # Lire la liste de coups attendus
    with open(liste_coups_corrects_txt, 'r') as fichier:
        liste_coups_reels = fichier.readlines()

    # Nettoyer les listes (supprimer les espaces blancs en début/fin de ligne)
    liste_coups_predits = [coup.strip() for coup in liste_coups_predits]
    liste_coups_reels = [coup.strip() for coup in liste_coups_reels]

    # Vérifier que les listes ont la même longueur
    assert len(liste_coups_predits) == len(liste_coups_reels), "Les listes de coups n'ont pas la même longueur."

    # Initialiser les compteurs
    coups_corrects = 0
    total_coups = len(liste_coups_reels)

    # Comparer les coups prédits aux coups réels
    for coups_reel, coups_pred in zip(liste_coups_reels, liste_coups_predits):
        if coups_reel == coups_pred:
            coups_corrects += 1

    # Calculer le taux de bonnes prédictions par coup
    taux_bonnes_predictions = (coups_corrects / total_coups) * 100

    return taux_bonnes_predictions

# Exemple d'utilisation
taux = precision_coup('Liste_coups2.txt', 'Feuille_176_verite_terrain.txt')
print(f"Taux de bonnes prédictions par coup : {taux:.0f}%")
