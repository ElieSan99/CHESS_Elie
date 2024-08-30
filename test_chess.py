# Import des modules nécessaires
import chess
import chess.pgn
import difflib


# Dictionnaire de traduction des pièces 

#FR->EN
dict_conversion_fr_en = {
    "C": "N", "F": "B", "R": "K", "D": "Q", "T": "R", "o-o":"O-O", "o-o-o":"O-O-O"
}

def conversion_piece(coup):
    # Traduire les pièces et les notations spéciales
    for fr, en in dict_conversion_fr_en.items():
        coup = coup.replace(fr, en)
    return coup

#EN->FR
dict_conversion_en_fr = {
    "N": "C", "B": "F", "R": "T", "Q": "D", "K": "R", "O-O":"o-o", "O-O-O":"o-o-o"
}

def conversion_piece_inverse(coup):
    # Traduire les pièces et les notations spéciales
    for en, fr in dict_conversion_en_fr.items():
        coup = coup.replace(en, fr)
    return coup


def lire_liste_coups(ficher):
    # Lecture du fichier contenaant la liste des coups
    with open(ficher, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # Extraction des coups
    coups = []
    for line in lines:
        parts = line.strip().split()
        for part in parts:
            if "." not in part:
                coups.append(conversion_piece(part))
    
    return coups

def transformation_coups_en_liste(coups):
    '''Fonction qui transforme une ligne de coups en une liste
    avec les deux coups. 
    
    '''
    coups_transformes = []
    parts = coups.strip().split()
    for part in parts:
        if "." not in part:
            coups_transformes.append(conversion_piece(part))
    
    return coups_transformes

def transformation_coups_en_txt(coups):
    '''Fonction qui transforme une liste de coups en une ligne pour le fichier txt 
    '''
    #TODO


def tester_coups(coups):
    '''Tester si les coups sont légaux ou pas'''

    # Initialisation de l'échiquier
    jeu = chess.Board()

    # Liste pour stocker les coups corrigés
    coups_corriges = []

    # Application des coups et affichage de la FEN après chaque cou
    for coup in coups:
        #print('moves possibles', jeu.legal_moves)
        print('coup prédit :', coup)
        coups_possibles = list(map(jeu.san, jeu.legal_moves))
        print('moves possibles', coups_possibles)
        if coup in coups_possibles:
            jeu.push_san(coup)
            coups_corriges.append(coup)
        else:
            similarites = []
            for coup2 in coups_possibles:
                similarite = difflib.SequenceMatcher(None, coup.lower(), coup2.lower()).ratio()
                similarites.append((similarite, coup2))
            # Trouver le coup le plus similaire
            coup_le_plus_similaire = max(similarites, key=lambda x: x[0])
            _, coup_suggere = coup_le_plus_similaire

            coups_corriges.append(coup_suggere)

    
    return coups_corriges


# générer le fichier txt de la liste des coups
def generer_liste_coups_txt(coups_corriges, fichier_sortie):
    with open(fichier_sortie, 'w', encoding='utf-8') as file:
        for i, coup in enumerate(coups_corriges):
            num_coup = i // 2 + 1  # Calcul du numéro de coup
            if i % 2 == 0:
                file.write(f"{num_coup}. {conversion_piece_inverse(coup)} ")
            else:
                file.write(f"{conversion_piece_inverse(coup)}\n")
                
            

# Test transformation_coups()
#coups = '1. d5 c5'
#coups_transformes = transformation_coups(coups)
#print(coups_transformes)

coups = lire_liste_coups('IMG_0177_verite_terrain.txt')
print(coups)
coups_corriges = tester_coups(coups)

generer_liste_coups_txt(coups_corriges, 'Liste_coups_corriges.txt')

