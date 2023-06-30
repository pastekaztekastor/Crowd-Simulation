# version CPU

## Compilation

Copiez intégralement le contenu du fichier `CMakeLists.txt` avec le code suivant :

```cmake
cmake_minimum_required(VERSION 3.0)
project(GPUSimulation VERSION 0.1.0 LANGUAGES CXX CUDA)

# Spécifier les fichiers source
set(SOURCES
    src/gpu/function.cpp
    src/gpu/kernel.cu
    src/gpu/main.cu
)

# Spécifier les fichiers d'en-tête
set(HEADERS
    src/gpu/kernel.hpp
    src/gpu/main.hpp
    src/gpu/utils/commonCUDA.hpp
)

# Spécifier le répertoire de sortie pour l'exécutable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/exe)

# Ajouter les options de compilation pour CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_30)

# Activer le langage C
enable_language(C)

# Inclure les bibliothèques HDF5
find_package(HDF5 REQUIRED COMPONENTS C HL)
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS})

# Compiler les fichiers source avec g++
add_executable(gpuSimulation ${SOURCES} ${HEADERS})

# Lier les librairies CUDA et HDF5
target_link_libraries(gpuSimulation ${CUDA_LIBRARIES} ${HDF5_LIBRARIES})
```

Ensuite, à la racine du projet, exécutez les commandes suivantes :

```bash
~ cmake .
~ make
```

Une fois compilé sans erreur, vous pouvez exécuter le programme avec la commande suivante :

```bash
~./exe/gpuSimulation
```

Assurez-vous de vous trouver dans le répertoire racine du projet lors de l'exécution des commandes.

## Utilisation

-nous avons eu une réunions avec les responsable de la partie maths

- donc pour la programmation avec GPU on vas utilisé CUDA
- La liste de paramètre de la simulaiton qui sont initialisé
  - `populationPosition` liste des individues
  - `populationIndex` liste des index de selection des individue
  - `map` la carte
  - `cost` la carte avec le couts ou les mure vallent NaN
  - `simExit` les cordonnée en x, y de la sortie.
  - `simDimX` et `simDimY` qui sont les évolutions de `_xParam` et `_yParam`
  - `simDimP` et `simDimG` qui sont les évolutions de `_nbIndividual` et `_nbGeneration`
  - `simPIn` le nombre de personne encore dans la simulation.
  - `settings` qui est un groupe de paramètre mais qui sont tous des variable unique
    - 0 `settings_model` avec le numéros de model
    - 1 `settings_dir` le chemin d'export
    - 2 `settings_dirName` le nom du dossier de l'export
    - 3 `settings_fileName` un préfixe au nom de chaque fichier
    - 4 `settings_exportType` type de l'export (png, jpg, txt, ...)
    - 5 `settings_exportFormat` ce qui est exporté (map, position, congestion, ...)
    - 6 `settings_debug` Affichage console pour le debug
    - 7 `settings_debugMap` Affichage console des maps
    - 8 `settings_finishCondition` On s'arrette à quel condition

- l'ordre des fonctions
  - initialisation
    - prise en compte des variables données par l'utilisateur
    - génération de la population
    - génération du tableau d'index
    - génération de la carte
    - génération de la carte des couts (option)
  - simulation
    - mélange des index :
      - shuffle
      - passage "avant/arrière"
    - utilisation du kernel/model
      - il retourne le déplacement de l'individue
      - choix selon le `settings_model`
      - le kernel modifie
        - la ``map``
        - la ``population``
        - ``simPIn``
    - Exorte la frame
    - Maybe progress bar
  - Retourne le temps total

### Help

- `-x` : Définit la dimension en x de la simulation (valeur attendue : entier).
- `-y` : Définit la dimension en y de la simulation (valeur attendue : entier).
- `-p` : Définit le nombre d'individus dans la simulation (valeur attendue : entier).
- `-g` : Définit le nombre de générations ou de frames de la simulation (valeur attendue : entier).
- `-debug` : Définit le niveau de débogage (valeurs attendues : "off", "all", "time", "normal").
- `-debugMap` : Active ou désactive l'affichage du mode débogage de la carte (valeurs attendues : "on", "off").
- `-model` : Définit le modèle de simulation à utiliser (valeurs attendues : de 0 à 7).
- `-exptT` : Définit le type d'exportation des résultats de la simulation (valeurs attendues : "txt", "jpeg", "bin", etc.).
- `-exptF` : Définit le format d'exportation des résultats de la simulation (valeurs attendues : "m", "p", "c").
- `-fCon` : Définit la condition de fin de la simulation (valeurs attendues : "fix", "empty").
- `-dir` : Définit le répertoire de destination des fichiers exportés (valeur attendue : chemin du répertoire).
- `-dirName` : Définit le nom personnalisé du répertoire de destination des fichiers exportés (valeur attendue : nom du répertoire).
- `-fileName` : Définit le nom personnalisé du fichier exporté (valeur attendue : nom du fichier).
- `-help`, `-h` : Affiche l'aide du programme.
Si l'option `-h` ou `-help` est spécifiée, le programme affiche cette liste d'arguments avec leurs descriptions, puis se termine.

Actuellement la seule utilitée du programme est de faire des additions parralélisé sur GPU.

## Gestion de la notion Atomic

- mettre dans le tab map les index our le humains, -1 vide, -2 sortie.
- Teste deplacement par atomic résultat
- restructurer les kernel avec un init, launch, clear

## Gestion des exports

Il y a deux types d'exportation possibles : des images/animations et un tableau d'évolution.

Pour le tableau d'évolution, il s'agit d'un tableau en 3D :

- L'axe x représente les individus.
- L'axe y représente les caractéristiques de chaque individu, telles que les coordonnées x et y ainsi que le temps d'attente.
- L'axe z représente chaque frame, et toutes les frames doivent être incluses dans le tableau.

Pour l'exportation vidéo :

- Tout d'abord, un fichier d'export est créé.
- Ensuite, le nombre de frames à inclure dans la vidéo est choisi (par exemple, 1 sur x frames).
- Pour les frames sélectionnées, une image du gif est capturée.
- Enfin, le gif est enregistré en tant que fichier de sortie.
