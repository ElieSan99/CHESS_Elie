def distance_similaire(str1, str2):
    m, n = len(str1), len(str2)
    
    # Initialisation de la matrice
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Remplissage de la première ligne et première colonne
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    # Remplissage de la matrice
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 1
            if str1[i - 1].lower() == str2[j - 1].lower():
                cost = 0.5  # Moindre coût pour une différence de casse
            elif str1[i - 1] == str2[j - 1]:
                cost = 0  # Aucun coût si les caractères sont identiques
            
            dp[i][j] = min(dp[i - 1][j] + 1,        # Suppression
                           dp[i][j - 1] + 1,        # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution

    return dp[m][n]

# Exemple d'utilisation
#candidat = "cf6"
#coup2 = "Cf6"
#coup3 = "Ff6"

#print(f"Distance entre '{candidat}' et '{coup2}': {distance_similaire(candidat, coup2)}")
#print(f"Distance entre '{candidat}' et '{coup3}': {distance_similaire(candidat, coup3)}")
