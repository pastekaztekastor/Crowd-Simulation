# Journal de bord

## 2023-06-01

- **Matin** : Présentation du projet (1h15)
- **Après-midi** : Départ vers 18:45
  - [x] Mise en ordre des informations présentées le matin (partie introduction).
  - [x] Lecture sur le sujet de la simulation de foule.
  - [x] Recherche documentaire variée.
  - [x] Compréhension du domaine.

## 2023-06-02

- **Matin** : Arrivée à 8h / Pause de 10:32->10:49 / fin à 12:18
  - [x] Terminer la version utilisant uniquement le CPU.
  - [x] Corriger les fautes dans les documents Markdown.
  - [x] Écrire les descriptions des fonctions.
  - [x] Terminer la fonction `generatePopulation` dans [onlyCPU.cpp](src/onlyCPU.cpp).
- **Après-midi** : 13:00 - 16:30
  - [x] Écrire la documentation pour la fonction `generatePopulation` dans [onlyCPU.md](markdown/onlyCPU.md).
  - [x] Écrire la fonction `generateMap` dans [onlyCPU.cpp](src/onlyCPU.cpp).
  - [x] Écrire la fonction `shuffleIndex` dans [onlyCPU.cpp](src/onlyCPU.cpp).
  - [ ] Écrire la fonction `shifting` dans [onlyCPU.cpp](src/onlyCPU.cpp).
  - [ ] Écrire la fonction `generatesJsonFile` dans [onlyCPU.cpp](src/onlyCPU.cpp).
  - [x] Écrire la fonction `printMap` dans [onlyCPU.cpp](src/onlyCPU.cpp).
  - J'ai bien avancé sur l'ensemble de l'écriture du programme CPU. J'ai commencé à faire des fonctions qui me serviront de toute façon lors de la parallélisation sur CUDA pour mieux voir ce qu'il se passe, comme `printMap`.
  - Je me suis rendu compte qu'il était impossible de se passer d'un tableau de déservant de carte pour pouvoir avoir une vision sur les voisins en direct sans avoir à parcourir tout le tableau des individus présents dans la simulation.
- **Soir** 18:00 - 19:30
  - Je rencontre un problème, soit dans l'affichage de la carte, soit dans sa génération.
  - ![image de preuve](content/Err002.png)
  - Je commence à me documenter sur l'affichage dynamique en ligne de commande dans un terminal. Je ne pense pas que ça soit vraiment utile.

## 2023-06-06

- **Matin** : Arrivée à 8:05 - 11:45
  - [x] Debug de la fonction `generateMap` dans [onlyCPU.cpp](src/onlyCPU.cpp).
  - [x] Debug de la fonction `printMap` dans [onlyCPU.cpp](src/onlyCPU.cpp).
  - [x] Retirer tous les passages de dimensions dans les fonctions pour simplifier le code, car je me suis rendu compte que je pouvais simplement me contenter de passer les tableaux et que la fonction `sizeof()` me donnait les dimensions.
  - Finalement non cette méthode ne foncitonne que pour les allocation non dynamique et je fait tout en dynamique.

## 2023-06-07

- **Matin** : Arrivée à 8:00 - 11:30
  - [x] Configuration du poste pour ne plus avoir à développer sur ma machine personnelle.
  - [x] Finalisation du module "Shifting".
  - Actuellement, la fonction "Shifting" ne prend pas en compte la sortie des individus. Lorsqu'une personne sort, elle n'est plus affichée, mais elle continue d'exister dans la simulation. Je suis confronté à la difficulté de gérer cette situation...
  
- **Après-midi** : 13:00 - 17:00
  - J'ai deux options lorsque l'individu quitte la simulation : soit le supprimer de la structure qui gère la population (ce qui réduit la consommation de mémoire), soit stocker une position prédéfinie (par exemple, (-1, -1)). J'ai opté pour la deuxième option, car elle est plus rapide à mettre en œuvre et ne nécessite pas l'ajout d'une notion d'identifiant dans le tableau de population.
  - [ ] Faire des paramètres argc argv plus propre.
  - Quand on recalcule le vecteur déplacement à chaque frame on peux pas afficher un déplacement linéaire. Donc on à la struct de position qui passe en float 
  - [x] Passer tout le code [onlyCPU](../src/onlyCPU.cpp) en ``float ** positions``
