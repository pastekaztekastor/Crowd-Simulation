# Exploration de `onlyCPU` 

Donc le but est de faire la même chose que l'objectif mais uniquement sur CPU pour bien comprendre le fonctionnement. De plus on pourra faire de Benchmark de performance de la parallélisation (un petit programme python fera le taff)

## Stucture du programme

Pour mieux comprendre le fonctionnement du programme voici un schéma de son fonctionnement  

```mermaid
---
title : fonctionnement de la fonction `main`
---
graph TB;
    a(initialisation Varible Globale)           --> b
    b(génération de la population)              --> c
    b                                           -->|Pour la config de départ| e
    c(mélangle la population)                   --> d
    d(calcul du déplacement)                    --> e
    e(génère un fichier json pour processing)   -->|Nb génération| c
```
