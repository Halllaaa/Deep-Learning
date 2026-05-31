# Partie I - MLP et ingenierie PyTorch

## 1. Objectif

L'objectif de cette partie est de traiter un probleme de classification supervisee sur des donnees tabulaires reelles avec un perceptron multicouche, appele MLP. Le travail permet aussi de montrer la maitrise des bases de PyTorch : `nn.Module`, `nn.Sequential`, les parametres, le calcul du gradient, le `state_dict`, la sauvegarde du modele et l'utilisation du CPU ou du GPU.

## 2. Dataset choisi

Le dataset utilise est Breast Cancer Wisconsin. Il contient des mesures numeriques calculees a partir d'images de cellules mammaires. L'objectif est de classer chaque exemple en deux classes : tumeur maligne ou tumeur benigne.

Ce dataset est pertinent pour cette partie car il s'agit de donnees tabulaires reelles, composees de variables numeriques. Chaque exemple est represente par un vecteur de caracteristiques, ce qui correspond bien a l'entree attendue par un MLP.

## 3. Preparation des donnees

Les donnees sont separees en trois ensembles :

- ensemble d'apprentissage;
- ensemble de validation;
- ensemble de test.

L'ensemble d'apprentissage sert a ajuster les poids du modele. L'ensemble de validation sert a comparer les variantes du modele pendant les experiences. L'ensemble de test est conserve pour l'evaluation finale.

Les variables numeriques sont normalisees avec `StandardScaler`. Cette etape est importante pour un MLP, car les reseaux de neurones sont sensibles a l'echelle des variables. Si certaines colonnes ont des valeurs beaucoup plus grandes que d'autres, elles peuvent dominer l'apprentissage.

## 4. Concepts PyTorch importants

### `nn.Module`

`nn.Module` est la classe de base pour construire un modele PyTorch. Un modele personnalise herite de cette classe et definit une methode `forward`. Cette methode decrit comment les donnees passent de l'entree vers la sortie.

### `nn.Sequential`

`nn.Sequential` permet d'empiler des couches dans un ordre simple. Il est pratique lorsque le modele est strictement lineaire dans son enchainement : couche, activation, couche, activation, sortie.

### Parametres

Les parametres sont les tenseurs entrainables du modele, principalement les poids et les biais des couches lineaires. PyTorch les detecte automatiquement lorsque les couches sont declarees dans un `nn.Module`.

### Gradient et retropropagation

Pendant l'apprentissage, le modele calcule une erreur avec une fonction de cout. La retropropagation calcule ensuite le gradient de cette erreur par rapport aux parametres. L'optimiseur utilise ces gradients pour modifier les poids du modele.

### `state_dict`

Le `state_dict` est un dictionnaire contenant les poids et biais du modele. Il permet de sauvegarder puis de recharger un modele entraine.

### Device

Le device indique si les calculs sont faits sur CPU ou GPU. En PyTorch, le modele et les donnees doivent etre places sur le meme device.

## 5. Architectures implementees

Deux versions du MLP sont implementees :

- une version avec `nn.Sequential`;
- une version avec une classe personnalisee heritant de `nn.Module`.

Les deux modeles utilisent des couches lineaires, des activations ReLU et une couche de sortie a deux neurones, car le probleme comporte deux classes.

## 6. Initialisation des poids

Trois strategies d'initialisation sont testees :

- initialisation gaussienne;
- initialisation constante;
- initialisation Xavier.

L'initialisation influence le debut de l'apprentissage. Une mauvaise initialisation peut ralentir l'optimisation ou rendre l'apprentissage instable. Xavier est souvent adaptee aux reseaux profonds car elle cherche a conserver une variance raisonnable des activations entre les couches.

## 7. Evaluation

Les performances sont evaluees avec les metriques suivantes :

- accuracy;
- precision;
- recall;
- F1-score;
- matrice de confusion.

L'accuracy mesure la proportion totale de predictions correctes. La precision indique parmi les exemples predits positifs combien sont reellement positifs. Le recall mesure parmi les exemples positifs reels combien sont detectes par le modele. Le F1-score combine precision et recall.

## 8. Analyse critique attendue

Un MLP est pertinent pour ce dataset car les donnees sont deja representees sous forme de vecteurs numeriques. Il peut apprendre des relations non lineaires entre les variables. Cependant, il ne possede pas de biais architectural specifique pour les donnees tabulaires. Il peut donc etre moins efficace que certains modeles classiques, comme les forets aleatoires ou le gradient boosting, surtout sur de petits datasets.

Ses limites principales sont :

- sensibilite a la normalisation des variables;
- besoin de regler plusieurs hyperparametres;
- risque de surapprentissage sur les petits jeux de donnees;
- interpretation moins directe que certains modeles statistiques ou arbres de decision.

## 9. Reponse de synthese - Partie I

Un MLP bien parametre peut constituer une solution pertinente pour la classification tabulaire lorsque les donnees sont numeriques, correctement normalisees et lorsque la relation entre les variables et la classe cible est non lineaire. Sur le dataset Breast Cancer Wisconsin, chaque observation est representee par un vecteur de caracteristiques, ce qui correspond directement au format d'entree d'un MLP.

La comparaison entre plusieurs initialisations permet de montrer que l'apprentissage ne depend pas seulement de l'architecture, mais aussi des conditions initiales de l'optimisation. L'initialisation constante est generalement moins interessante, car elle peut limiter la diversite des neurones au debut de l'apprentissage. Les initialisations gaussienne et Xavier sont plus adaptees, Xavier etant souvent plus stable.

Cependant, le MLP a des limites. Il ne tient pas compte d'une structure particuliere autre que le vecteur de caracteristiques. Sur des donnees tabulaires, la performance depend fortement de la qualite du preprocessing, du choix des hyperparametres et de la taille du dataset. Ainsi, meme si le MLP peut obtenir de bonnes performances, son utilisation doit etre justifiee par une analyse experimentale et comparee de maniere critique.

Big Picture 
The whole script follows this logic: 
Load dataset
-> split train/validation/test
-> normalize features
-> convert to PyTorch tensors
-> build MLP models
-> initialize weights
-> train models
-> validate during training
-> evaluate on test set
-> save metrics, plots, and models
-> reload best model to prove persistence
 This is exactly what part 1 asks for: 
 Theory, PyTorch implementation, intialization comparaison, device handling, save/reload, and classification metrics.