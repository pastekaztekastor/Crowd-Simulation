# README - Projet Simulaiton de foule

Ce README présente la version la plus aboutie du programme sur la simulation de foule. Bien que cette version actuelle fonctionne exclusivement sur CPU, le code peut être adapté pour prendre en charge une version GPU (cette adaptation est expliquée en détail dans un autre fichier).

L'objectif de ce document est de guider les utilisateurs dans l'utilisation du programme. Nous aborderons les points suivants :

## Installation Préalable

Avant de commencer, assurez-vous de disposer des éléments suivants :

- [CLion](https://www.jetbrains.com/clion/): Un environnement de développement intégré (IDE) qui facilite la programmation en C++.
- [OpenCV](https://opencv.org/): Une bibliothèque open-source de vision par ordinateur et de traitement d'image.
- [FFmpeg](https://www.ffmpeg.org/): Un outil puissant pour enregistrer, convertir et diffuser des vidéos et des flux audio.

Veuillez installer ces éléments en suivant les instructions fournies sur leurs sites respectifs. Une fois ces installations préalables effectuées, vous serez prêt à configurer et exécuter le programme **SimulationDeFoule**. Assurez-vous de les avoir correctement installés avant de poursuivre.

Il est important de noter que dans de nombreux cas, **FFmpeg** est installé conjointement avec **OpenCV**. En effet, **OpenCV** utilise souvent les fonctionnalités de **FFmpeg** pour le traitement des flux vidéo et audio. Par conséquent, si vous installez **OpenCV** en suivant les instructions fournies, il est probable que vous disposiez déjà de **FFmpeg** sur votre système. Cela simplifie grandement le processus d'installation, car vous n'avez généralement pas besoin de configurer **FFmpeg** séparément. Cependant, il est toujours conseillé de vérifier la documentation spécifique de votre distribution **OpenCV** pour confirmer la disponibilité de **FFmpeg** et ses dépendances.

Ensuite, ouvrez le dépôt Git dans **CLion**.

>
> Cependant, il est important de noter que si vous choisissez d'utiliser un autre environnement de développement intégré (IDE) que **CLion**, des complications pourraient survenir. Les paramètres de configuration, les chemins d'accès aux dépendances et autres configurations spécifiques peuvent différer entre les IDE. Par conséquent, bien que notre documentation soit basée sur l'utilisation de **CLion**, vous pourriez rencontrer des défis supplémentaires si vous optez pour un IDE différent. Nous recommandons vivement l'utilisation de **CLion** pour garantir une expérience de développement fluide et cohérente avec le projet **SimulationDeFoule**.
>

## Arborescence du Projet

Pour faciliter la navigation et l'utilisation du projet **SimulationDeFoule**, voici les deux éléments clés que les utilisateurs doivent connaître :

1. **Dossier des Images d'Entrée:** [ici](../../input Image) Toutes les images d'entrée nécessaires au fonctionnement du programme doivent être placées dans ce dossier spécifique. Assurez-vous que les images sont correctement nommées et formatées conformément aux directives de la documentation.

2. **Fichier de Configuration:** [ici](utils/setup.hpp) Le fichier de configuration est un élément essentiel pour personnaliser le comportement du programme. Ce fichier contient des paramètres et des valeurs qui influencent les résultats de la simulation. Il est important de le modifier en conséquence pour obtenir les résultats souhaités.

3. **Dossier des Fichiers Temporaires**: [ici](../../tmp/frames) Le programme génère des fichiers temporaires pour chaque frame de calcul. Ces fichiers sont stockés dans ce dossier désigné. Ils sont cruciaux pour le déroulement efficace de la simulation, mais vous pouvez les supprimer en toute sécurité après utilisation
4. **Dossier de la Vidéo Générée**: [ici](../../video) Une fois la simulation effectuée avec succès, la vidéo résultante sera placée dans ce dossier spécifique. Vous y trouverez la vidéo finale qui rassemble les résultats de la simulation.

## Fichiers d'Entrée
Pour que le programme fonctionne correctement, veuillez préparer les fichiers d'entrée requis. Ces fichiers sont essentiels pour le bon déroulement du processus et doivent être formatés conformément aux instructions fournies dans la documentation.

### Contraintes sur les Images d'Entrée

L'utilisation d'images d'entrée appropriées est cruciale pour la bonne exécution du programme **SimulationDeFoule**. Voici les contraintes à respecter pour les images d'entrée :

1. **Dimensions Uniformes:** Toutes les images d'entrée doivent avoir les mêmes dimensions. Cela garantit la cohérence du traitement et de la simulation.

2. **Nommage Précis:** Les images d'entrée doivent être nommées conformément aux spécifications suivantes :
  - La carte/topologie de la simulation doit être nommée "map.png". Il ne peut y en avoir qu'une seule. Les murs de la carte doivent être de couleur noir absolu (255, 255, 255).
  - Les images de population doivent être nommées "population0.png", "population1.png", et ainsi de suite pour chaque population. Les pixels noirs indiquent une probabilité d'apparition de 0%, tandis que les pixels blancs indiquent une probabilité de 100%. Les teintes entre noir et blanc représentent un gradient de probabilité.
  - Les images de sortie doivent être nommées "exit0.png", "exit1.png", et ainsi de suite pour chaque sortie. Les sorties doivent être de couleur verte absolue (0, 255, 0).

3. **Dossier de Stockage:** Les images d'entrée doivent être organisées dans un dossier spécifique. Assurez-vous de ranger toutes les images dans ce dossier pour une référence facile lors de l'exécution du programme.

4. **Équivalence Population-Sortie:** Il est impératif d'avoir autant d'images de population que d'images de sortie. Chaque image de population correspond à une image de sortie spécifique, assurant ainsi une corrélation entre les différentes entités du système simulé.

En respectant rigoureusement ces contraintes, vous vous assurez que le programme fonctionne correctement et produit des résultats cohérents lors de la simulation.

### Organisation des Images d'Entrée

Les images d'entrée jouent un rôle crucial dans la simulation du projet **SimulationDeFoule**. Pour assurer une organisation claire et précise, suivez ces étapes :

1. Utilisez le dossier existant "Input Images" pour stocker toutes les images d'entrée nécessaires pour vos simulations.

2. Pour chaque simulation que vous souhaitez exécuter, créez un sous-dossier à l'intérieur du dossier "Input Images". Vous pouvez nommer ce sous-dossier comme vous le souhaitez, par exemple, "Simulation_A", "Projet_B", etc.

3. À l'intérieur de chaque sous-dossier de simulation, placez les images nécessaires selon les contraintes précédemment mentionnées :
  - La carte/topologie de la simulation doit être nommée "map.png".
  - Les images de population doivent être nommées "population0.png", "population1.png", etc.
  - Les images de sortie doivent être nommées "exit0.png", "exit1.png", etc.

4. Pour chaque simulation, modifiez le fichier de configuration existant. Dans ce fichier de configuration, indiquez le nom du sous-dossier de simulation que vous avez créé. Cela informe le programme de l'emplacement des images spécifiques pour cette simulation.

En respectant cette structure, vous pourrez facilement gérer et référencer les images d'entrée pour chaque simulation. Par exemple, si vous créez 5 populations, assurez-vous d'avoir 5 images de population, 5 images de sortie et 1 image de carte, pour un total de 11 fichiers dans le sous-dossier de simulation.

### Configuration des Paramètres

Pour personnaliser le comportement du programme **SimulationDeFoule**, vous pouvez modifier le fichier de configuration [setup.hpp](utils/setup.hpp) qui est un fichier d'en-tête C. Voici une explication des différents paramètres présents dans ce fichier :

```c
#define __NB_POPULATION__               1
```
Ce paramètre détermine le nombre de populations simulées. Vous pouvez ajuster cette valeur en fonction du nombre de populations que vous souhaitez inclure dans votre simulation.

```c
#define __NAME_DIR_INPUT_FILES__        "Simulation_A"
```
Ce paramètre spécifie le nom du sous-dossier de simulation que vous souhaitez utiliser. Remplacez `"Simulation_A"` par le nom du sous-dossier que vous avez créé. Assurez-vous que le nom correspond exactement au nom du sous-dossier.

```c
#define __PRINT_DEBUG__                 false
```
Lorsque ce paramètre est défini sur `true`, des messages de débogage seront affichés pendant l'exécution du programme. Cela peut vous aider à comprendre le comportement du programme lors de la phase de développement. Si vous souhaitez désactiver ces messages de débogage, définissez ce paramètre sur `false`.

```c
#define __VIDEO_FPS__                   120
```
Ce paramètre définit le nombre d'images par seconde (FPS) pour la génération de la vidéo finale. Vous pouvez ajuster cette valeur en fonction de la fluidité souhaitée pour la vidéo générée.

```c
#define __NB_FRAMES_MAX__               -1
```
Ce paramètre contrôle le nombre maximum de frames à simuler. Si vous le laissez à la valeur par défaut `-1`, le nombre de frames sera déterminé automatiquement en fonction de la configuration de la simulation. Si vous souhaitez limiter manuellement le nombre de frames, remplacez `-1` par la valeur souhaitée.

Après avoir modifié ces paramètres en fonction de vos besoins, n'oubliez pas de sauvegarder les modifications. Cela garantira que la simulation se déroule conformément à vos préférences spécifiques.

## Configuration Initiale

Si vous devez modifier le fichier cmake pour une raison quelconque, il est conseillé de créer une copie préalablement, de sorte que vous puissiez restaurer cette copie modifiée en cas de besoin. La voici :

```cmake
cmake_minimum_required(VERSION 3.0)
project(GPUObjSimulation VERSION 0.1.0 LANGUAGES CXX)

# Définir le standard C++11
set(CMAKE_CXX_STANDARD 11)

# Définir la macro GRAPHICS_CARD_PRESENT si la carte graphique est présente
# Trouver CUDA
find_package(CUDA QUIET)

if (CUDA_FOUND)
  set(GRAPHICS_CARD_PRESENT 1)
  find_package(CUDA REQUIRED)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_30)
else()
  set(GRAPHICS_CARD_PRESENT 0)
endif()
# Afficher le résultat
message(STATUS "CUDA found: ${CUDA_FOUND}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

# Spécifier les fichiers source
if (CUDA_FOUND)
  set(SOURCES
          src/gpuObjectVersion/main.cpp

          src/gpuObjectVersion/utils/utils.cpp

          src/gpuObjectVersion/Kernel.cu
          src/gpuObjectVersion/Map.cpp
          src/gpuObjectVersion/Population.cpp
          src/gpuObjectVersion/Settings.cpp
          src/gpuObjectVersion/Simulation.cpp
          src/gpuObjectVersion/Export.cpp)
else()
  set(SOURCES
          src/gpuObjectVersion/main.cpp

          src/gpuObjectVersion/utils/utils.cpp
          src/gpuObjectVersion/utils/matrixMove.cpp

          src/gpuObjectVersion/Kernel.cpp
          src/gpuObjectVersion/Map.cpp
          src/gpuObjectVersion/Population.cpp
          src/gpuObjectVersion/Settings.cpp
          src/gpuObjectVersion/Simulation.cpp
          src/gpuObjectVersion/Export.cpp)
endif()


# Spécifier les fichiers d'en-tête
set(HEADERS
        src/gpuObjectVersion/utils/utils.hpp

        src/gpuObjectVersion/Kernel.hpp
        src/gpuObjectVersion/Map.hpp
        src/gpuObjectVersion/Population.hpp
        src/gpuObjectVersion/Settings.hpp
        src/gpuObjectVersion/Simulation.hpp
        src/gpuObjectVersion/Export.hpp
        src/gpuObjectVersion/utils/matrixMove.cpp)

# Spécifier le répertoire de sortie pour l'exécutable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/exe)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add OpenCV include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Compiler les fichiers source avec g++
add_executable(GPUObjSimulation ${SOURCES} ${HEADERS})

if (CUDA_FOUND)
  target_link_libraries(GPUObjSimulation PRIVATE ${CUDA_LIBRARIES})
endif()

target_link_libraries(GPUObjSimulation PRIVATE ${OpenCV_LIBS})
# Link your executable with OpenCV libraries
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

ATTENTION : ces commande ne sont pas à prendre en compte si vous êtes sous CLion ! Uniquement si vous êtes en ligne de commande.

## Exécution du Programme
Pour lancer le programme **SimulationDeFoule**, suivez ces étapes simples :

1. Ouvrez le projet **SimulationDeFoule** dans **CLion** en sélectionnant le dossier du projet.
2. Naviguez jusqu'au fichier contenant le code principal du programme (généralement `main.cpp` ou similaire).
3. Dans **CLion**, cliquez sur le bouton "Exécuter" (généralement représenté par un triangle vert) situé dans la barre d'outils ou appuyez sur la combinaison de touches appropriée pour lancer le programme.

Après avoir suivi ces étapes, **CLion** compilera et exécutera le programme **SimulationDeFoule**. Assurez-vous de consulter la sortie de la console pour surveiller le déroulement de la simulation et les éventuels messages de débogage ou d'erreur.

Voici un exemple de sortie console plus détaillée que vous pourriez obtenir en exécutant le programme **SimulationDeFoule** :

```
---------- Crowd Simulation ----------
NOMBRE DE PASSE CARTE COUT : 5
 - Nb mure : 9527
 - Nb population : 1
 - Nb individu total : 4660
 - Frames max : 2609600000
DÉBUT DE LA SIMULATION 
[=========================================================] 100.00% : 4820 Frames
CRÉATION DE LA VIDÉO 
[=========================================================] 100.00% : 4821 Frames
RÉSUMÉ 
 - calculé en 4820 frames 
 - temps pour l'initialisation 0.990729 sec 
 - temps pour la simulation 1.17562 min 
 - temps pour l'export vidéo 1.69686 min 

Process finished with exit code 0
```

Cette sortie console fournit un aperçu plus détaillé de la simulation. Elle inclut des informations telles que le nombre de passes de la carte, le nombre de murs, le nombre total d'individus, le nombre de frames maximales, et d'autres détails liés au déroulement de la simulation.

Continuez à surveiller la sortie console pour obtenir des informations détaillées sur les différentes étapes de la simulation, ainsi que sur les temps d'initialisation, de simulation et d'export vidéo.

Cela vous permettra de mieux comprendre le comportement du programme et de surveiller la progression de la simulation.

Si la simulation prend trop de temps pour s'exécuter et que vous souhaitez interrompre le processus en cours, vous pouvez simplement "tuer" le processus. Voici comment vous pourriez procéder :

1. Revenez à l'interface de la console où vous avez lancé la simulation.

2. Appuyez sur la combinaison de touches **Ctrl+C** (ou **Cmd+C** sur macOS). Cela enverra un signal d'interruption au processus en cours d'exécution.

3. Le programme sera interrompu et la console affichera un message indiquant que le processus a été arrêté.

Ou alors en cliquant sur l'icon stop dans la bare d'icon en haut.

Si vous choisissez de tuer le processus de cette manière, la simulation s'arrêtera à l'état où elle en était au moment de l'interruption. Cependant, gardez à l'esprit que la vidéo ne sera pas générée.

## Fichiers Générés

Après avoir effectué la simulation dans le projet **SimulationDeFoule**, plusieurs fichiers générés fournissent un aperçu complet du processus :

1. **Vidéo de Simulation :** Une vidéo qui illustre le déroulement de la simulation.

2. **Dossier des Frames :** Ce dossier contient les frames individuelles de la simulation.

  - **Fichiers TXT par Frame :** À l'intérieur du dossier, chaque fichier TXT correspondant à une frame contient les informations suivantes : `[indice population] [indice individu] [position en x] [position en y] [position en z]`. Veuillez noter que la position en z est réservée pour une fonction future mais n'est pas actuellement implémentée.

Ces fichiers vous permettent de visualiser la simulation à travers la vidéo et de consulter les données de position des individus à chaque instant grâce aux fichiers TXT associés à chaque frame.

## Débogage et Manipulation
En cas de problèmes ou d'erreurs lors de l'exécution, consultez la section de débogage de la documentation. Vous y trouverez des conseils sur la résolution des problèmes courants et des astuces pour manipuler efficacement le programme.

