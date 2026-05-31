# Rapport Final - Projet Deep Learning

## Conception, implementation, comparaison et analyse critique de modeles de deep learning pour donnees tabulaires, images et sequences

### Module : Deep Learning

### Dataset tabulaire : Breast Cancer Wisconsin

### Dataset image : Fashion-MNIST

### Dataset sequentiel : IMDb

---

# 1. Introduction

Le deep learning repose sur un principe general : apprendre automatiquement une fonction de prediction a partir de donnees. Cependant, toutes les donnees n'ont pas la meme structure. Une ligne de donnees tabulaires, une image et un texte ne possedent pas la meme geometrie ni les memes dependances internes. C'est pourquoi un meme paradigme d'apprentissage supervise doit etre adapte a la nature des donnees et a leur organisation.

Ce projet etudie trois familles de modeles :

- le MLP pour les donnees tabulaires;
- le CNN pour les images;
- les modeles recurrentes RNN, LSTM, GRU et Seq2Seq pour les sequences textuelles.

L'objectif est de montrer que le choix d'une architecture n'est pas arbitraire. Il depend de la structure statistique et geometrique du dataset. Le projet combine donc une etude theorique, une implementation PyTorch, des experiences comparatives et une analyse critique des resultats.

---

# 2. Methodologie generale

Les trois parties suivent une demarche experimentale commune :

1. choix d'un dataset reel adapte au type de donnees;
2. preparation des donnees;
3. implementation de plusieurs modeles sous PyTorch;
4. entrainement et validation;
5. evaluation sur un ensemble de test;
6. comparaison des resultats;
7. interpretation critique.

Les experiences ont ete concues pour etre executables sur un ordinateur portable. Pour cette raison, certaines parties, notamment Fashion-MNIST et IMDb, utilisent des sous-ensembles reduits. Cette contrainte est prise en compte dans l'interpretation des resultats.

---

# 3. Partie I - MLP pour donnees tabulaires

## 3.1 Objectif

La premiere partie traite un probleme de classification supervisee sur donnees tabulaires avec un perceptron multicouche, ou MLP. Elle permet aussi d'etudier les bases de PyTorch : `nn.Module`, `nn.Sequential`, les parametres, les gradients, le `state_dict`, l'initialisation, la sauvegarde et le rechargement d'un modele.

## 3.2 Dataset

Le dataset utilise est Breast Cancer Wisconsin. Il contient :

```text
569 echantillons tabulaires
30 caracteristiques numeriques par echantillon
2 classes : malignant et benign
```

La repartition utilisee est :

```text
397 echantillons d'apprentissage
86 echantillons de validation
86 echantillons de test
```

Chaque echantillon correspond a une observation medicale de cellule tumorale. L'objectif est de predire si la tumeur est maligne ou benigne.

## 3.3 Preparation des donnees

Les donnees sont separees en apprentissage, validation et test. La separation est stratifiee afin de conserver une proportion similaire des classes dans chaque sous-ensemble.

Les caracteristiques sont normalisees avec `StandardScaler`. Cette etape est importante car les MLP sont sensibles a l'echelle des variables. Le scaler est ajuste uniquement sur les donnees d'apprentissage, puis applique aux donnees de validation et de test afin d'eviter toute fuite d'information.

## 3.4 Architectures implementees

Deux versions du MLP ont ete implementees :

- une version avec `nn.Sequential`;
- une version avec une classe personnalisee heritant de `nn.Module`.

Les deux architectures utilisent des couches lineaires, des activations ReLU, du dropout et une couche de sortie a deux neurones.

La version `nn.Sequential` est concise et adaptee aux architectures lineaires simples. La version personnalisee avec `nn.Module` offre plus de flexibilite et permet de definir explicitement la methode `forward`.

## 3.5 Initialisation et sauvegarde

Trois strategies d'initialisation ont ete testees :

- initialisation gaussienne;
- initialisation constante;
- initialisation Xavier.

L'initialisation constante est moins favorable car elle donne des poids trop similaires aux neurones. Les initialisations gaussienne et Xavier introduisent une diversite initiale plus utile pour l'apprentissage.

Le meilleur modele est sauvegarde avec son `state_dict`, puis recharge afin de verifier la reproductibilite et la persistance du modele.

## 3.6 Resultats

| Modele | Initialisation | Accuracy test | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Custom MLP | Gaussienne | 0.9419 | 0.9804 | 0.9259 | 0.9524 |
| Sequential MLP | Xavier | 0.9419 | 0.9804 | 0.9259 | 0.9524 |
| Custom MLP | Xavier | 0.9302 | 0.9615 | 0.9259 | 0.9434 |
| Sequential MLP | Gaussienne | 0.9302 | 0.9615 | 0.9259 | 0.9434 |
| Sequential MLP | Constante | 0.9186 | 0.9796 | 0.8889 | 0.9320 |
| Custom MLP | Constante | 0.9186 | 0.9796 | 0.8889 | 0.9320 |

Les meilleurs resultats sont obtenus par le MLP personnalise avec initialisation gaussienne et par le MLP `nn.Sequential` avec initialisation Xavier. Les deux atteignent une accuracy de 94.19 % et un F1-score de 95.24 %.

## 3.7 Interpretation precise des metriques et figures

L'accuracy de 94.19 % signifie que le meilleur modele classe correctement environ 94 % des observations de l'ensemble de test. Comme le test contient 86 observations, cela correspond approximativement a 81 predictions correctes. Cette metrique donne une vision globale de la performance, mais elle doit etre completee par la precision, le recall et le F1-score.

La precision de 0.9804 indique que, parmi les observations predites comme appartenant a la classe positive, la tres grande majorite sont correctes. Le recall de 0.9259 indique que le modele retrouve environ 92.59 % des exemples positifs reels. Le F1-score de 0.9524, qui combine precision et recall, confirme que le meilleur MLP presente un bon equilibre entre detection correcte et limitation des fausses predictions.

Le tableau comparatif montre aussi l'influence de l'initialisation. Les initialisations gaussienne et Xavier donnent les meilleurs resultats, tandis que l'initialisation constante obtient une performance plus faible. Ce resultat est coherent avec la theorie : si plusieurs neurones commencent avec des poids identiques ou tres similaires, ils risquent d'apprendre des representations redondantes.

Les courbes d'apprentissage sauvegardees dans `outputs/figures` permettent d'observer l'evolution de la loss d'apprentissage, de la loss de validation et de l'accuracy de validation. Lorsque la loss diminue et que l'accuracy de validation augmente, cela indique que le modele apprend effectivement une relation entre les caracteristiques et la classe cible. Si la loss d'apprentissage diminuait fortement tandis que la loss de validation augmentait, cela indiquerait un surapprentissage. Dans cette experience, les courbes montrent globalement une convergence rapide, ce qui est coherent avec la petite taille et la relative simplicite du dataset.

Les matrices de confusion permettent d'identifier les erreurs par classe. Dans un contexte medical, cette analyse est importante car une confusion entre une tumeur maligne et une tumeur benigne n'a pas la meme consequence pratique que d'autres erreurs de classification. La matrice de confusion complete donc les metriques globales en montrant la nature des erreurs et non seulement leur nombre.

## 3.8 Analyse critique

Le MLP est pertinent pour ce dataset car les donnees sont deja representees sous forme de vecteurs numeriques. Apres normalisation, le modele peut apprendre des relations non lineaires entre les caracteristiques et la classe cible.

Cependant, le MLP presente des limites. Il ne possede pas de biais architectural specifique aux donnees tabulaires. Il peut aussi etre sensible au choix des hyperparametres, a la taille du dataset et au preprocessing. Sur certains datasets tabulaires, des methodes classiques comme les forets aleatoires ou le gradient boosting peuvent etre competitives.

## 3.9 Reponse de synthese - Partie I

Un MLP bien parametre constitue une solution pertinente pour la classification tabulaire lorsque les donnees sont numeriques, normalisees et lorsque la relation entre variables et classes est non lineaire. Sur Breast Cancer Wisconsin, les performances obtenues montrent que le MLP apprend efficacement a separer les classes maligne et benigne.

Ses principales limites concernent sa dependance au preprocessing, son risque de surapprentissage sur de petits datasets et son manque d'interpretabilite directe. L'analyse experimentale montre aussi que l'initialisation influence la qualite finale de l'apprentissage.

---

# 4. Partie II - CNN pour classification d'images

## 4.1 Objectif

La deuxieme partie etudie les reseaux de neurones convolutionnels pour la classification d'images. L'objectif est de montrer pourquoi un CNN est mieux adapte qu'un MLP aux donnees visuelles.

## 4.2 Dataset

Le dataset utilise est Fashion-MNIST. Il contient des images en niveaux de gris representant dix categories de vetements et accessoires.

Le dataset complet contient :

```text
60 000 images d'apprentissage
10 000 images de test
70 000 images au total
```

Pour une execution rapide sur ordinateur portable, l'experience utilise :

```text
4 000 images d'apprentissage
1 000 images de validation
1 000 images de test
```

Chaque image a la forme :

```text
1 x 28 x 28
```

Les dix classes sont : T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag et Ankle boot.

## 4.3 Fondements des CNN

Un MLP applique aux images doit d'abord aplatir l'image en un vecteur. Une image 28 x 28 devient donc un vecteur de 784 valeurs. Cette operation detruit la structure spatiale : le modele ne sait plus naturellement quels pixels etaient voisins.

Un CNN conserve la structure de grille de l'image. Il repose sur trois idees principales :

- la localite : les pixels proches sont lies;
- le partage des poids : le meme filtre est applique sur toute l'image;
- la hierarchie des representations : les premieres couches detectent des motifs simples, les couches plus profondes construisent des motifs plus complexes.

## 4.4 Operations manuelles

Le projet implemente manuellement :

- la correlation croisee 2D;
- le max-pooling;
- l'average-pooling.

Ces implementations sont comparees avec les operations PyTorch equivalentes. La taille de sortie d'une convolution est calculee avec :

```text
sortie = floor((entree + 2 * padding - kernel) / stride) + 1
```

Exemples :

```text
28x28, kernel=5, padding=0, stride=1 -> 24x24
28x28, kernel=5, padding=2, stride=1 -> 28x28
28x28, kernel=3, padding=1, stride=2 -> 14x14
```

## 4.5 Architectures comparees

Six modeles ont ete testes :

- MLP sur images aplaties;
- CNN inspire de LeNet avec max-pooling;
- CNN avec average-pooling;
- CNN avec plus de filtres;
- CNN avec convolution 1x1;
- CNN avec convolutions strided.

Ces variantes permettent d'etudier l'influence du padding, du stride, du pooling, du nombre de filtres et de la convolution 1x1.

## 4.6 Resultats

| Modele | Accuracy test | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| CNN strided | 0.8190 | 0.8259 | 0.8227 | 0.8220 |
| MLP images aplaties | 0.8050 | 0.8093 | 0.8079 | 0.8059 |
| CNN LeNet max-pooling | 0.7940 | 0.8174 | 0.7985 | 0.8006 |
| CNN avec 1x1 | 0.7910 | 0.8076 | 0.7964 | 0.7982 |
| CNN plus de filtres | 0.7730 | 0.7933 | 0.7802 | 0.7816 |
| CNN average-pooling | 0.7650 | 0.7779 | 0.7666 | 0.7663 |

Le meilleur modele est le CNN strided, avec une accuracy de 81.90 % et un F1 macro de 82.20 %.

## 4.7 Interpretation precise des metriques et figures

L'accuracy de 81.90 % du CNN strided signifie que le modele classe correctement environ 819 images sur les 1 000 images de test utilisees. Le F1 macro de 0.8220 est particulierement utile car Fashion-MNIST comporte dix classes. Contrairement a l'accuracy, le F1 macro donne le meme poids a chaque classe, ce qui permet de mieux verifier si le modele fonctionne de maniere equilibree sur l'ensemble des categories.

La comparaison avec le MLP sur images aplaties est centrale. Le MLP obtient une accuracy de 80.50 %, tandis que le meilleur CNN obtient 81.90 %. La difference n'est pas tres grande, mais elle va dans le sens attendu : le CNN exploite mieux la structure spatiale de l'image. Le fait que l'ecart soit modere s'explique par le nombre limite d'images, le faible nombre d'epochs et la simplicite volontaire des architectures.

Le tableau montre aussi que tous les choix architecturaux n'ameliorent pas automatiquement les performances. Le CNN avec plus de filtres n'est pas le meilleur dans cette experience. Cela indique qu'augmenter la capacite d'un modele ne suffit pas toujours : il faut aussi tenir compte du dataset, du temps d'entrainement, du risque de surapprentissage et de la strategie de reduction spatiale.

Les courbes d'entrainement sauvegardees dans `outputs/figures` permettent de verifier si le modele apprend progressivement. Pour les CNN, une baisse de la loss et une augmentation de l'accuracy de validation indiquent que les filtres convolutifs apprennent des representations utiles. Si la validation stagne alors que la loss d'apprentissage diminue, cela peut indiquer que le modele apprend trop specifiquement les images d'apprentissage.

Les matrices de confusion sont essentielles pour Fashion-MNIST car certaines classes sont visuellement proches. Par exemple, Shirt, Pullover et Coat peuvent etre confondus car leurs silhouettes se ressemblent. A l'inverse, Trouser, Bag, Sandal ou Ankle boot sont souvent plus faciles a reconnaitre car leur forme est plus distinctive. L'interpretation de la matrice de confusion permet donc de relier les erreurs du modele a la nature visuelle des classes.

Les cartes de caracteristiques visualisees montrent comment un CNN transforme une image brute en representations internes. Les premieres cartes peuvent mettre en evidence des contours, des zones sombres, des transitions de texture ou des formes locales. Ces figures ne permettent pas de donner une interpretation parfaite de chaque neurone, mais elles montrent que le CNN ne travaille pas seulement sur des pixels isoles : il construit progressivement des representations visuelles.

## 4.8 Analyse critique

Le CNN strided obtient le meilleur score dans cette experience. Cela confirme l'interet des convolutions pour exploiter la structure spatiale des images. Le MLP reste competitif, mais il ignore explicitement les relations locales entre pixels.

La difference entre MLP et CNN n'est pas tres grande dans cette experience car le nombre d'images et d'epochs est limite. Avec le dataset complet et un entrainement plus long, les CNN ont generalement un avantage plus net.

Les classes visuellement proches, comme Shirt, Pullover et Coat, sont plus difficiles a separer. Cela apparait dans les matrices de confusion.

## 4.9 Reponse de synthese - Partie II

Un CNN est plus pertinent qu'un MLP pour une tache de classification d'images car il respecte la geometrie de l'image. Les filtres convolutifs capturent des motifs locaux, le partage des poids reduit le nombre de parametres, et les couches successives construisent des representations de plus en plus abstraites.

Le padding controle la conservation des dimensions, le stride reduit la taille spatiale, le pooling compresse l'information et le nombre de filtres augmente la capacite du modele. Les experiences montrent que ces choix influencent directement les performances et doivent etre justifies empiriquement.

---

# 5. Partie III - RNN, LSTM, GRU et Seq2Seq sur IMDb

## 5.1 Objectif

La troisieme partie etudie les donnees sequentielles a travers un corpus textuel reel : IMDb. Le texte est une sequence de tokens, et l'ordre des tokens porte une information essentielle. Cette partie implemente des modeles de langage RNN, LSTM et GRU, puis un mini systeme Seq2Seq.

## 5.2 Dataset

Le dataset IMDb complet contient :

```text
25 000 critiques d'apprentissage
25 000 critiques de test
50 000 critiques au total
```

Pour une execution raisonnable sur CPU, l'experience utilise :

```text
600 critiques d'apprentissage
100 critiques de validation
200 critiques de test
```

Le vocabulaire construit contient :

```text
2 500 tokens
```

Les tokens speciaux utilises sont `<pad>`, `<unk>`, `<sos>` et `<eos>`.

## 5.3 Tokenisation et vocabulaire

La tokenisation transforme les critiques IMDb en sequences de tokens. Les mots sont mis en minuscules, les balises HTML simples sont nettoyees et les mots, nombres et ponctuations principales sont extraits.

Chaque token est ensuite converti en identifiant numerique grace au vocabulaire. Les mots absents du vocabulaire sont remplaces par `<unk>`.

## 5.4 Modele de langage

Un modele de langage apprend a predire le prochain token a partir des tokens precedents. La probabilite d'une sequence est factorisee selon la regle de chaine :

```text
P(w1, ..., wn) = P(w1) * P(w2|w1) * ... * P(wn|w1, ..., w(n-1))
```

La perplexite est utilisee comme metrique principale. Une perplexite plus faible indique un modele moins incertain et donc meilleur.

## 5.5 RNN, LSTM, GRU et clipping

Le RNN simple maintient un etat cache qui resume le contexte precedent. Il est simple mais peut souffrir de gradients instables ou d'une mauvaise memorisation des dependances longues.

Le LSTM ajoute une cellule memoire et des portes pour mieux controler l'information conservee ou oubliee.

Le GRU utilise aussi des portes, mais avec une structure plus simple que le LSTM.

Le gradient clipping limite la norme des gradients afin de reduire les explosions de gradients pendant la retropropagation a travers le temps, ou BPTT.

## 5.6 Resultats des modeles de langage

| Modele | Clipping | Test loss | Test perplexity | Norme max gradient |
|---|---:|---:|---:|---:|
| RNN clipped | 1.0 | 7.2427 | 1397.93 | 0.7443 |
| GRU clipped | 1.0 | 7.3869 | 1614.65 | 0.4697 |
| LSTM clipped | 1.0 | 7.5367 | 1875.72 | 0.3771 |
| RNN no clipping | None | 7.5802 | 1959.10 | 0.5151 |

Dans cette experience reduite, le RNN avec clipping obtient la plus faible perplexite. Cela ne signifie pas que le RNN est toujours superieur au LSTM ou au GRU. Le resultat depend du faible nombre d'epochs, de la taille limitee du corpus et de la longueur courte des sequences. En general, sur des corpus plus grands et des dependances plus longues, LSTM et GRU peuvent mieux exploiter leur mecanisme de portes.

## 5.7 Interpretation precise des metriques et figures des modeles de langage

La loss mesure l'erreur moyenne du modele lorsqu'il predit le prochain token. Une loss plus faible indique que le modele attribue une probabilite plus elevee aux tokens corrects. La perplexite est derivee de cette loss et mesure l'incertitude du modele. Une perplexite de 1397.93 signifie que le modele reste tres incertain, ce qui est normal dans cette experience reduite : le vocabulaire contient 2 500 tokens, les critiques IMDb sont variees et l'entrainement est volontairement court.

Le RNN avec clipping obtient la meilleure perplexite dans cette configuration. La comparaison avec le RNN sans clipping montre l'interet du gradient clipping : la perplexite passe de 1959.10 a 1397.93. Cela suggere que limiter la norme des gradients aide le modele a apprendre de facon plus stable, meme si les gradients observes ne sont pas extremement eleves.

Les resultats du LSTM et du GRU sont moins bons dans cette experience courte. Cette observation doit etre interpretee avec prudence. LSTM et GRU possedent plus de mecanismes internes que le RNN simple, mais ils peuvent avoir besoin de plus de donnees ou de plus d'epochs pour exploiter pleinement leur capacite de memorisation. Le resultat experimental doit donc etre relie au protocole utilise, et non generalise abusivement.

La figure `part3_language_model_curves.png` montre l'evolution de la loss de validation et de la perplexite. Elle permet de verifier si les modeles progressent pendant l'entrainement. Une diminution de la perplexite indique que le modele devient moins incertain dans la prediction du prochain token. Cette figure est directement liee a l'objectif de la partie III : comparer la capacite des architectures recurrentes a modeliser une sequence textuelle.

## 5.8 Seq2Seq

Un modele Seq2Seq transforme une sequence d'entree en sequence de sortie. Il comporte :

- un encodeur, qui lit la sequence source;
- un decodeur, qui produit la sequence cible.

Comme le dataset choisi est IMDb, le Seq2Seq a ete applique a une tache de reconstruction de courts extraits de critiques. L'encodeur lit un extrait, puis le decodeur tente de reconstruire le meme extrait.

Pendant l'entrainement, le modele utilise le teacher forcing : le decodeur recoit le vrai token precedent au lieu de sa propre prediction precedente.

Deux strategies de decodage sont comparees :

- decodage glouton;
- beam search.

## 5.9 Resultats Seq2Seq

| Modele | Score type BLEU glouton | Score type BLEU beam search |
|---|---:|---:|
| Seq2Seq GRU reconstruction | 0.0972 | 0.0960 |

Les scores sont faibles. Cela montre que, meme si la loss diminue pendant l'entrainement, la generation libre reste difficile. Le modele tend a produire des tokens frequents et ne reconstruit pas encore correctement les sequences.

## 5.10 Interpretation precise des resultats Seq2Seq

Le score de type BLEU compare les tokens generes avec les tokens de reference. Un score proche de 1 indiquerait une reconstruction tres proche de la reference. Les scores obtenus, autour de 0.097, montrent que le modele ne reconstruit pas encore correctement les extraits IMDb en decodage libre.

La difference entre decodage glouton et beam search est tres faible. Cela signifie que, dans cette configuration, le beam search n'apporte pas d'amelioration notable. Une raison possible est que le modele lui-meme n'a pas encore appris une distribution de sortie suffisamment riche : garder plusieurs hypotheses ne suffit pas si les probabilites produites par le decodeur restent pauvres ou concentrees sur quelques tokens frequents.

La figure `part3_seq2seq_training_curves.png` montre cependant que la loss d'apprentissage et la loss de validation diminuent au fil des epochs. Cela indique que le modele apprend bien la tache sous teacher forcing. La difficulte apparait surtout au moment du decodage libre, lorsque le decodeur doit utiliser ses propres predictions precedentes. Cette difference entre entrainement et generation est un point important dans les architectures Seq2Seq.

Les exemples sauvegardes dans `part3_seq2seq_predictions.txt` confirment cette limite. Les sequences generees sont courtes et repetitives. Cela montre que le modele encodeur-decodeur implemente les principes demandes, mais que ses performances restent limitees par la taille du corpus, la simplicite de l'architecture et l'absence d'attention.

## 5.11 Analyse critique

La partie sequence montre la difficulte du texte par rapport aux donnees tabulaires et aux images. Les modeles recurrentes doivent apprendre a la fois le vocabulaire, l'ordre des tokens et le contexte. La perplexite reste elevee car le corpus utilise est reduit, le vocabulaire est limite et l'entrainement est court.

Le Seq2Seq implemente bien les concepts demandes, mais ses performances sont limitees. Une amelioration naturelle serait d'utiliser plus de donnees, plus d'epochs, un mecanisme d'attention, ou une architecture de type Transformer.

## 5.12 Reponse de synthese - Partie III

Les architectures recurrentes permettent de modeliser efficacement une sequence car elles traitent les tokens dans l'ordre et maintiennent un etat cache representant le contexte. Le passage d'un RNN simple vers un LSTM ou un GRU est justifie par la necessite de mieux gerer les dependances longues et la stabilite de l'apprentissage.

Le schema encodeur-decodeur devient pertinent lorsque l'objectif est de produire une sequence complete a partir d'une autre sequence. Le decodage glouton est simple, tandis que le beam search explore plusieurs hypotheses. Les resultats obtenus montrent cependant que la generation de texte reste une tache difficile avec un modele petit et un corpus reduit.

---

# 6. Discussion transversale finale

La problematique centrale du projet est la suivante :

```text
Comment le deep learning adapte-t-il ses architectures a la structure des donnees tabulaires, images et sequentielles ?
```

Les trois parties montrent que le choix de l'architecture depend directement de la structure des donnees.

## 6.1 Donnees tabulaires et MLP

Les donnees tabulaires sont representees sous forme de vecteurs de caracteristiques. Chaque ligne correspond a un echantillon, et chaque colonne correspond a une variable. Dans ce contexte, le MLP est adapte car il prend naturellement un vecteur en entree et apprend des combinaisons non lineaires entre les variables.

Cependant, le MLP ne suppose pas de structure particuliere entre les colonnes. Il traite les caracteristiques comme un vecteur global. Cela convient a Breast Cancer Wisconsin, mais peut etre limite lorsque les interactions entre variables sont complexes ou lorsque le dataset est petit.

## 6.2 Images et CNN

Une image est une grille spatiale. Les pixels proches sont fortement lies. Un MLP, en aplatissant l'image, perd cette structure spatiale. Le CNN est plus adapte car il utilise des filtres locaux, partage les poids et construit des representations hierarchiques.

Ainsi, le CNN introduit un biais architectural coherent avec la geometrie de l'image. Il apprend des motifs locaux comme les bords, puis les combine en formes plus complexes.

## 6.3 Sequences et modeles recurrentes

Un texte est une sequence ordonnee. Le sens depend de l'ordre des tokens et du contexte precedent. Les RNN, LSTM et GRU sont adaptes car ils maintiennent un etat cache au fil de la sequence. Les modeles Seq2Seq ajoutent une structure encodeur-decodeur pour produire une sequence de sortie complete.

Contrairement aux donnees tabulaires, ou chaque exemple est un vecteur fixe, une sequence peut avoir une longueur variable et contenir des dependances longues. Cela rend l'apprentissage plus difficile.

## 6.4 Meme paradigme, architectures differentes

Dans les trois cas, le paradigme general est supervise :

```text
prediction -> calcul de la loss -> backpropagation -> mise a jour des poids
```

Mais l'architecture change :

| Type de donnees | Structure | Architecture adaptee | Raison |
|---|---|---|---|
| Tabulaire | Vecteur de caracteristiques | MLP | Apprentissage de combinaisons non lineaires |
| Image | Grille spatiale | CNN | Localite, partage des poids, hierarchie visuelle |
| Texte/sequence | Ordre temporel ou linguistique | RNN, LSTM, GRU, Seq2Seq | Memoire du contexte et generation sequentielle |

Le deep learning ne se resume donc pas a empiler des couches. Il consiste a choisir une architecture dont les hypotheses correspondent a la structure des donnees.

---

# 7. Limites generales du projet

Les experiences ont ete adaptees pour etre executables sur ordinateur portable. Cela implique plusieurs limites :

- les sous-ensembles Fashion-MNIST et IMDb sont reduits;
- les modeles sont volontairement petits;
- le nombre d'epochs est limite;
- le Seq2Seq n'utilise pas d'attention;
- aucune recherche exhaustive d'hyperparametres n'a ete effectuee.

Ces limites ne remettent pas en cause l'objectif pedagogique du projet. Elles doivent toutefois etre mentionnees pour interpreter correctement les resultats.

---

# 8. Conclusion

Ce projet montre que les architectures de deep learning doivent etre adaptees a la structure des donnees. Le MLP est pertinent pour les donnees tabulaires car il traite naturellement des vecteurs de caracteristiques. Le CNN est plus adapte aux images car il exploite la localite et la structure spatiale. Les RNN, LSTM, GRU et Seq2Seq sont necessaires pour les sequences car ils prennent en compte l'ordre et le contexte.

Les resultats experimentaux confirment cette logique generale. Le MLP obtient de bonnes performances sur Breast Cancer Wisconsin, le CNN strided obtient la meilleure performance sur Fashion-MNIST, et les modeles recurrentes permettent de construire une premiere modelisation du texte IMDb. Les limites observees, notamment sur Seq2Seq, montrent aussi qu'une architecture correcte ne suffit pas : la taille du corpus, la duree d'entrainement, le vocabulaire et les mecanismes avances comme l'attention influencent fortement la qualite finale.

Ainsi, le projet met en evidence le lien essentiel entre theorie, implementation et analyse critique : comprendre la structure des donnees est indispensable pour choisir, entrainer et interpreter un modele de deep learning.
