# Approximation de Surface par MLP : L'Explorateur d'Île 
## Nom Complet : Mecheter Ines

## Présentation du Projet
Ce projet implémente un **Perceptron Multi-Couches (MLP)** à partir de zéro (from scratch) en utilisant **NumPy** pour résoudre un problème de régression complexe. L'objectif est de simuler un "explorateur" qui doit redessiner la carte topographique d'une île à partir de points d'altitude dispersés.

Le modèle apprend à approximer la fonction : $f: \mathbb{R}^2 \to \mathbb{R}$, où $(x, y)$ représentent les coordonnées géographiques et $z$ représente l'altitude.

##  Objectif Mathématique
La fonction d'altitude cible est définie par la surface non-linéaire suivante :
$$f(x, y) = \sin(\sqrt{x^2 + y^2}) + 0.5 \cos(2x + 2y)$$

##  Caractéristiques Techniques
- **Implémentation Vectorisée :** Respect strict des contraintes du calcul matriciel. Aucune boucle `for` n'est utilisée pour traiter les échantillons, garantissant une performance optimale avec NumPy.
- **Rétropropagation Manuelle :** Implémentation complète des gradients et de la mise à jour des poids via la règle de dérivation en chaîne (Chain Rule).
- **Architecture Modulaire :** Structure par défaut [2, 64, 64, 1] optimisée pour l'ajustement de surfaces non-linéaires.
- **Normalisation des Données :** Utilisation du Min-Max Scaling pour les cibles et normalisation standard pour les entrées afin d'assurer la stabilité du gradient.

##  Installation et Utilisation

1. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   python tp.py
   ```
