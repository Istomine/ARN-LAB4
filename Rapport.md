Lab 04: Deep Neural Networks
Shyshmarov Alexandre - Pinto da Cunha da Mata Guilherme

# 2. Digit recogni on from raw data

Select a final model (e.g., the one with best performance) for analysis (please, see below the summary of work to know what to include in the report)

```
Pour mon model final j'ai modifié par rapport au model de base :
    La structure de réseau de neuron. J'ai ajouté une deuxième couche cachée, j'ai changé le nombre de neuron (il est a 30 maintenant) et la fonction d'activation, c'est la relu maintenant

    J'ai modifié le nombre d'epoch. On est a 10 alors qu'avant, on était a 3
```

1. What is the learning algorithm being used to optimize the weights of the neural
networks?

```
RMSprop.
Etant donnée que l'algo n'a pas de parametre il utlise les parametre par défaut :
learning_rate = 0.001 / rho = 0.9 / epsilon = 1e-7 / etc..

La loss fonction est categorical_crossentropy
```
2. For each experiment excepted the last one (shallow network learning from raw data, shallow network learning from features and CNN)

    1. Select a neural network topology and describe the inputs, indicate how many are they, and how many outputs?
    2. Compute the number of weights of each model (e.g., how many weights between the input and the hidden layer, how many weights between each pair of layers, biases, etc..) and explain how do you get to the total number of weights.
```
Pour le model que j'ai sélectionner avant :

Il y a 784 inputs ce qui correspond à une image 28x28. Il y a 30 neurones sur la couche input + les biais de chaque neurone. On est à 23550 paramètres.
La seconde couche a 30 neurones et a une fonction d'activation relu. Avec 30 entré par neurones et 30 neurones au total on est à 900 paramètres + 30 biais.
La couche output a 10 neurones qui correspond aux 10 chiffres qu'on veut prédire. Et a softmax comme fonction d'activation. Chaque neurone sur cette couche a 30 entrées ce qui fait 300 paramètres + 10 biais.

Le total : Input = 30 * 784 + 30 / Hidden = 30 * 30 + 30 / Output = 30 * 10 + 10
```
3. Test at least three different meaningful cases
    
```
Quand je change le nombre d'input à 3, le model confond :
5 avec 0 / 5 avec 8 / 8 avec 2 / 8 avec 5 / 9 avec 7
```


![example1](images/2.3-input3.png)
```
La courbe est pas trop mal, on peut également voir qu'on a une loss qui est plutôt grande. Ce qui est assez mauvais pour un modèle.
```

![example2](images/2.3-input3Matr.png)
```
La matrice de confusion reflète ce qui a été dit plus haut. Certains nombres ont du mal a été reconnu par notre modèle.
```

```
Quand j'augment le nombre de couche caché à 3 couches , le model confond :
3 avec 9 / 8 avec 2 / 8 avec 3 / 9 avec 7
```
![example3](images/2.3-coucheCache.png)

```
On peut voir que le modèle est meilleur que celui avant, mais on voit que partir de 7 epoch on commence à voir de l'overfitting. Au niveau de la loss, elle est bien mieux qu'avant et l'accuracy est presque à 1.
```

![example4](images/2.3-coucheCacheematr.png)

```
La matrice de confusion est presque parfaite, si ce ne sont quelques chiffres qui posent toujours problème.
```

```
Quand j'augmente le nombre d'epoch a 40 , le model confond :
1 avec 6 / 2 avec 8 / 3 avec 5 / 9 avec 4 
```

![example3](images/2.3-epoch40.png)

```
On a très clairement un gros problème d'overfitting dès le début.
```

![example3](images/2.3-epoch40Matr.png)

```
Malgré l'overfitting vu juste avant, la matrice est plutôt pas mal. Mais si, on test le modèle sur une autre dataset, on aurait sûrement des résultats un peu moins bien.
```

# 3. Digit recogni on from features of the input data

# 4. Convolu onal neural network digit recogni on

# 5. Chest X-ray to detect pneumonia
