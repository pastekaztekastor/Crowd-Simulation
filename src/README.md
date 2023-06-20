# Source des fichiers de codes

Dans ce projet, nous avons développé deux versions de programmes pour démontrer le concept de parallélisation sur carte graphique :

1. Le programme `gpu` est conçu pour s'exécuter sur une carte graphique compatible avec CUDA, ce qui permet d'exploiter la puissance de calcul parallèle de la carte. Ce programme utilise des techniques de programmation CUDA pour répartir les tâches sur les threads GPU. Cependant, il offre également une option pour s'exécuter en mode CPU uniquement, pour les utilisateurs qui ne disposent pas d'une carte graphique compatible CUDA.

2. Le programme `cpu` est une version proof of concept qui fonctionne exclusivement sur le CPU. Il est conçu pour illustrer le fonctionnement du programme sans utiliser la parallélisation sur carte graphique. Bien qu'il puisse être moins performant que la version GPU, il permet aux utilisateurs de tester le programme et de comprendre son fonctionnement sans avoir besoin d'une carte graphique compatible CUDA.

Ces deux versions de programme offrent une approche flexible pour exécuter le projet, en fonction des ressources matérielles disponibles. Les utilisateurs peuvent choisir d'exécuter le programme en utilisant la puissance de calcul parallèle de la carte graphique ou en utilisant uniquement le CPU.