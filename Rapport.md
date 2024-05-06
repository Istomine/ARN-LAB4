Lab 04: Deep Neural Networks
Shyshmarov Alexandre - Pinto da Cunha da Mata Guilherme

# 2. Digit recogni on from raw data

Pour mon model final j'ai modifié par rapport au model de base :
    La structure de réseau de neuron. J'ai ajouté une deuxième couche cachée, j'ai changé le nombre de neuron (il est a 30 maintenant) et la fonction d'activation, c'est la relu maintenant

    J'ai modifié le nombre d'epoch. On est a 10 alors qu'avant, on était a 3

1. What is the learning algorithm being used to optimize the weights of the neural
networks?
    RMSprop.
    Etant donnée que l'algo n'a pas de parametre il utlise les parametre par défaut :
    learning_rate = 0.001 / rho = 0.9 / epsilon = 1e-7 / etc..

    La loss fonction est categorical_crossentropy

2. 

    1. Select a neural network topology and describe the inputs, indicate how many are they, and how many outputs?
        Il y a 784 inputs ce qui correspond a une image 28x28. Il y a 30 neurons sur la couche input + les biais de chaque neurons. On est a 23550 parametres 

        La seconde couche a 30 neurons et a une fonction d'activation relu. Avec 30 entré par neurons et 30 neurons au total on est a 900 parametre + 30 biais

        la couche output a 10 neurons qui correponde aux 10 chiffres qu'on veut predire. Et a softmax comme fonction d'activation. Chaque neuron sur cette couche a 30 entrées ce qui fait 300 paramètres + 10 biais.

Le total : Input = 30 * 784 + 30 / Hidden = 30 * 30 + 30 / Output = 30 * 10 + 10

    3. 5 avec 0 / 5 avec 8 / 8 avec 2 / 8 avec 5 / 9 avec 7 quand je change le nombre de neuron d'input a 3
        3 avec 9 / 8 avec 2 / 8 avec 3 / 9 avec 7 quand j'augmente le nombre de couche caché a 3 couche
        1 avec 6 / 2 avec 8 / 3 avec 5 / 9 avec 4 quand j'augmente le nombre d'epoch a 40 (on a de l'overfitting')

# 3. Digit recogni on from features of the input data

# 4. Convolu onal neural network digit recogni on

# 5. Chest X-ray to detect pneumonia
