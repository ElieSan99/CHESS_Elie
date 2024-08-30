import chess

# Créer une instance d'échiquier
board = chess.Board()

# Extraire la liste des coups légaux
legal_moves = list(board.legal_moves)

# Convertir les coups en notations lisibles
legal_moves_str = [board.san(move) for move in legal_moves]

# Enregistrer la liste dans un fichier texte
with open("liste_coups_possibles.txt", "w") as file:
    for move in legal_moves_str:
        file.write(move + "\n")

print("Liste des coups possibles enregistrée dans 'liste_coups_possibles.txt'")
