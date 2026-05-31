# Partie II - CNN et vision par ordinateur

## 1. Objectif

L'objectif de cette partie est d'etudier la classification d'images avec des reseaux de neurones convolutionnels, appeles CNN. Contrairement a un MLP, un CNN conserve la structure spatiale de l'image et applique des filtres locaux pour extraire des caracteristiques visuelles.

Le dataset choisi est Fashion-MNIST. Il contient des images en niveaux de gris de taille 28 x 28 representant dix types de vetements ou accessoires : T-shirt, pantalon, pull, robe, manteau, sandale, chemise, basket, sac et bottine.

## 2. Pourquoi un MLP est limite pour les images

Un MLP transforme l'image en un long vecteur. Pour une image Fashion-MNIST, cela donne 28 x 28 = 784 valeurs. Cette transformation detruit l'organisation spatiale : deux pixels voisins dans l'image peuvent devenir de simples positions dans un vecteur.

Le probleme est que les images possedent une structure locale. Les pixels proches forment des bords, des textures et des formes. Un MLP n'utilise pas explicitement cette information. Il apprend une connexion complete entre tous les pixels et les neurones, ce qui augmente le nombre de parametres et ignore la geometrie de l'image.

## 3. Idees fondamentales des CNN

Un CNN repose sur trois idees principales.

La localite signifie qu'un filtre observe seulement une petite region de l'image a la fois. Par exemple, un filtre 5 x 5 analyse des petits blocs de pixels.

Le partage des poids signifie que le meme filtre est applique partout dans l'image. Cela reduit fortement le nombre de parametres et permet de detecter le meme motif a plusieurs positions.

La hierarchie des representations signifie que les premieres couches apprennent des motifs simples, comme des bords, tandis que les couches plus profondes apprennent des formes plus complexes.

## 4. Correlation croisee 2D

Dans un CNN, l'operation appelee convolution dans les bibliotheques est souvent une correlation croisee. Un noyau, ou filtre, glisse sur l'image. A chaque position, on multiplie les valeurs du patch par les valeurs du filtre, puis on additionne le resultat.

Si l'entree a une taille `n`, le noyau une taille `k`, le padding `p` et le stride `s`, la taille de sortie est :

```text
sortie = floor((n + 2p - k) / s) + 1
```

Exemples pour une image 28 x 28 :

- noyau 5, padding 0, stride 1 : sortie 24 x 24;
- noyau 5, padding 2, stride 1 : sortie 28 x 28;
- noyau 3, padding 1, stride 2 : sortie 14 x 14.

## 5. Padding, stride et pooling

Le padding ajoute des pixels autour de l'image. Il permet de mieux traiter les bords et peut conserver la taille spatiale.

Le stride controle le deplacement du filtre. Un stride plus grand reduit plus vite la taille de sortie, mais peut perdre des details.

Le pooling reduit la dimension spatiale. Le max-pooling garde la valeur maximale d'une region et met en avant les activations fortes. L'average-pooling calcule une moyenne et produit une representation plus lisse.

## 6. Convolution 1 x 1

Une convolution 1 x 1 ne melange pas les pixels voisins spatialement. Elle agit surtout sur les canaux. Elle permet de recombiner les cartes de caracteristiques, de modifier le nombre de canaux et d'ajouter de la non-linearite avec peu de cout.

## 7. Modeles implementes

Le script implemente plusieurs modeles :

- un MLP sur images aplaties;
- un CNN inspire de LeNet avec max-pooling;
- une variante avec average-pooling;
- une variante avec plus de filtres;
- une variante avec convolution 1 x 1;
- une variante utilisant des convolutions avec stride.

Cette comparaison permet d'analyser l'influence de plusieurs choix architecturaux : padding, stride, pooling, nombre de filtres et convolution 1 x 1.

## 8. Evaluation

Les modeles sont evalues avec :

- accuracy;
- precision macro;
- recall macro;
- F1-score macro;
- matrice de confusion.

La moyenne macro donne le meme poids a chaque classe, ce qui est utile pour comparer les performances entre les dix categories.

## 9. Interpretation des cartes de caracteristiques

Les cartes de caracteristiques visualisent les activations internes d'un CNN. Elles permettent d'observer ce que les filtres detectent dans une image. Les premieres cartes peuvent reagir a des contours, des zones sombres, des transitions ou des formes simples.

Ces visualisations ne prouvent pas exactement ce que le modele comprend, mais elles aident a interpreter le passage de l'image brute vers une representation interne plus abstraite.

## 10. Reponse de synthese - Partie II

Un CNN est plus pertinent qu'un MLP pour la classification d'images car il respecte la structure spatiale des donnees. Une image n'est pas seulement un vecteur de pixels independants : elle contient des relations locales entre pixels voisins. Les CNN exploitent cette structure grace aux filtres convolutifs, au partage des poids et a la construction progressive de representations.

Le padding influence la conservation des dimensions et la prise en compte des bords. Le stride controle la reduction spatiale et peut accelerer le calcul, mais un stride trop grand risque de supprimer des informations utiles. Le pooling reduit la taille des cartes de caracteristiques et rend le modele plus robuste a de petites variations. Le nombre de filtres controle la capacite du modele : plus de filtres permettent d'apprendre plus de motifs, mais augmentent le cout de calcul et le risque de surapprentissage.

La comparaison experimentale entre MLP et CNN permet de verifier que le CNN obtient generalement de meilleures performances sur des images, car son architecture correspond mieux a la geometrie des donnees. Les cartes de caracteristiques montrent enfin que le modele ne travaille pas seulement sur des pixels bruts, mais construit progressivement des representations visuelles internes.

