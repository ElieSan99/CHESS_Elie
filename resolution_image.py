import os
from PIL import Image
from shutil import copy2

# Définir les répertoires de base
rep = 'dataset'
new_rep = 'dataset2'

# Définir les sous-répertoires
sous_reps = ['train', 'val', 'test']

# Créer le nouveau répertoire de base s'il n'existe pas
os.makedirs(new_rep, exist_ok=True)

# Fonction pour vérifier la résolution de l'image et copier si 128x128
def verifier_et_copier_image(chemin_image, nouveau_chemin_image):
    try:
        with Image.open(chemin_image) as img:
            if img.size == (128, 128):
                # Créer les répertoires nécessaires dans le nouveau répertoire
                os.makedirs(os.path.dirname(nouveau_chemin_image), exist_ok=True)
                # Copier l'image
                copy2(chemin_image, nouveau_chemin_image)
                print(f"Copié {chemin_image} vers {nouveau_chemin_image}")
    except Exception as e:
        print(f"Erreur lors du traitement de {chemin_image}: {e}")

# Parcourir la structure des répertoires
for sous_rep in sous_reps:
    chemin_sous_rep = os.path.join(rep, sous_rep)
    nouveau_chemin_sous_rep = os.path.join(new_rep, sous_rep)
    for categorie in os.listdir(chemin_sous_rep):
        chemin_categorie = os.path.join(chemin_sous_rep, categorie)
        nouveau_chemin_categorie = os.path.join(nouveau_chemin_sous_rep, categorie)
        if os.path.isdir(chemin_categorie):
            for nom_image in os.listdir(chemin_categorie):
                chemin_image = os.path.join(chemin_categorie, nom_image)
                nouveau_chemin_image = os.path.join(nouveau_chemin_categorie, nom_image)
                verifier_et_copier_image(chemin_image, nouveau_chemin_image)

print("Traitement terminé.")
