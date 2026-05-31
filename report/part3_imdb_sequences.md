# Partie III - RNN, LSTM, GRU et Seq2Seq sur IMDb

## 1. Objectif

L'objectif de cette partie est d'etudier les modeles de deep learning adaptes aux donnees sequentielles. Contrairement aux donnees tabulaires et aux images, un texte est une suite ordonnee de tokens. L'ordre des mots change le sens, donc le modele doit tenir compte du contexte precedent.

Le dataset choisi est IMDb. Il contient des critiques de films ecrites en langage naturel. Dans ce projet, IMDb est utilise comme corpus textuel reel pour deux taches :

- modelisation du langage avec RNN, LSTM et GRU;
- mini systeme Seq2Seq de reconstruction de courts extraits de critiques.

IMDb est souvent utilise pour la classification de sentiment, mais ici nous l'utilisons surtout comme corpus sequentiel reel, car la consigne demande des modeles RNN, LSTM, GRU et Seq2Seq.

## 2. Pourquoi le texte est une donnee sequentielle

Dans un texte, les tokens sont ordonnes. Par exemple :

```text
this movie was good
```

n'a pas le meme sens que :

```text
good movie was this
```

Un modele de sequence doit donc traiter les elements dans leur ordre. C'est la raison pour laquelle les architectures recurrentes sont importantes.

## 3. Tokenisation

La tokenisation transforme un texte en une liste de tokens.

Exemple :

```text
This movie was excellent!
```

devient :

```text
["this", "movie", "was", "excellent", "!"]
```

Dans le script, une tokenisation simple est utilisee avec une expression reguliere. Les mots sont mis en minuscules, et certains signes de ponctuation sont gardes.

## 4. Vocabulaire

Le vocabulaire associe chaque token a un identifiant numerique.

Exemple :

```text
"movie" -> 45
"good" -> 88
"bad" -> 103
```

Les reseaux de neurones ne lisent pas directement les mots. Ils lisent des nombres. Le vocabulaire sert donc de traduction entre texte et entiers.

Le script utilise aussi des tokens speciaux :

```text
<pad> : complete les sequences trop courtes
<unk> : represente les mots inconnus
<sos> : debut de sequence pour le decodeur
<eos> : fin de sequence
```

## 5. Modele de langage

Un modele de langage apprend a predire le prochain token.

Exemple :

```text
this movie was
```

Le modele peut predire :

```text
good
```

L'objectif probabiliste est :

```text
P(w1, w2, ..., wn) = P(w1) * P(w2|w1) * P(w3|w1,w2) * ... * P(wn|w1,...,w(n-1))
```

Cette formule signifie que la probabilite d'une sequence complete peut etre decomposee en predictions successives du prochain token.

## 6. Perplexite

La perplexite mesure l'incertitude d'un modele de langage.

Une perplexite faible signifie que le modele predit mieux les prochains tokens.

Une perplexite elevee signifie que le modele est plus incertain.

Dans le rapport, on peut interpreter la perplexite comme le nombre moyen de choix plausibles que le modele considere a chaque prediction.

## 7. RNN simple

Un RNN traite une sequence token par token. Il garde un etat cache qui resume le contexte deja lu.

Forme simplifiee :

```text
h_t = f(x_t, h_(t-1))
```

`h_t` est l'etat cache au temps `t`.

Le probleme d'un RNN simple est qu'il peut avoir du mal a memoriser des dependances longues. Les gradients peuvent devenir trop petits ou trop grands pendant la retropropagation dans le temps.

## 8. BPTT

BPTT signifie Backpropagation Through Time.

C'est la retropropagation appliquee a une sequence. Le modele est comme "deplie" dans le temps, puis les gradients sont calcules a travers les etapes successives.

Si la sequence est longue, l'apprentissage peut devenir instable. C'est pour cela qu'on utilise souvent des sequences tronquees et du gradient clipping.

## 9. Gradient clipping

Le gradient clipping limite la norme des gradients.

Dans le script :

```python
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Si les gradients deviennent trop grands, ils sont reduits. Cela aide a eviter les explosions de gradients, surtout avec les RNN.

Le script compare :

```text
RNN sans clipping
RNN avec clipping
LSTM avec clipping
GRU avec clipping
```

## 10. LSTM

LSTM signifie Long Short-Term Memory.

Un LSTM ajoute une cellule memoire et des portes. Ces portes controlent ce que le modele garde, oublie et transmet.

Les portes principales sont :

```text
forget gate
input gate
output gate
```

Cela permet au LSTM de mieux gerer les dependances longues qu'un RNN simple.

## 11. GRU

GRU signifie Gated Recurrent Unit.

Un GRU est aussi un modele recurrent avec des portes, mais il est plus simple qu'un LSTM.

Il utilise principalement :

```text
update gate
reset gate
```

Le GRU est souvent plus rapide que le LSTM tout en gardant une bonne capacite de memorisation.

## 12. Comparaison RNN, LSTM, GRU

Le RNN simple est le plus basique. Il est rapide, mais il a souvent des difficultes avec les longs contextes.

Le LSTM est plus puissant pour garder de l'information sur une longue distance, mais il a plus de parametres.

Le GRU est un compromis : plus simple que le LSTM, mais plus stable qu'un RNN simple.

Dans le projet, on compare ces modeles avec :

- la loss;
- la perplexite;
- la stabilite des gradients;
- le temps d'entrainement.

## 13. Seq2Seq

Seq2Seq signifie sequence-to-sequence.

Un modele Seq2Seq transforme une sequence d'entree en une sequence de sortie.

Exemples classiques :

```text
traduction automatique
resume automatique
generation de texte
correction de texte
```

Comme l'utilisateur a demande IMDb, le script construit une tache de reconstruction de courts extraits de critiques IMDb. L'encodeur lit un extrait, puis le decodeur essaie de le reconstruire.

Ce n'est pas une traduction entre deux langues, mais c'est bien une tache sequence-vers-sequence sur un corpus textuel reel.

## 14. Encodeur et decodeur

L'encodeur lit la sequence source et produit un etat cache final.

Le decodeur utilise cet etat pour produire la sequence de sortie token par token.

Structure :

```text
source tokens -> encoder -> hidden state -> decoder -> output tokens
```

## 15. Teacher forcing

Pendant l'entrainement Seq2Seq, on donne au decodeur les vrais tokens precedents.

Exemple :

Pour predire le token 3, le decodeur recoit le vrai token 2 au lieu de sa propre prediction.

Cela rend l'apprentissage plus stable.

Dans notre implementation, les entrees du decodeur sont :

```text
<sos> + sequence cible decalee
```

## 16. Decodage glouton

Le decodage glouton choisit a chaque etape le token le plus probable.

Avantage :

```text
simple et rapide
```

Limite :

```text
un mauvais choix au debut peut deteriorer toute la sequence
```

## 17. Beam search

Beam search garde plusieurs hypotheses a chaque etape.

Au lieu de garder seulement la meilleure prediction, il garde par exemple les 3 meilleures sequences candidates.

Avantage :

```text
meilleure exploration des sorties possibles
```

Limite :

```text
plus couteux que le decodage glouton
```

## 18. Evaluation Seq2Seq

Pour la tache Seq2Seq, le script calcule un score de type BLEU simplifie.

BLEU compare la sequence generee avec la sequence reference en regardant les mots communs et les petits groupes de mots communs.

Comme la tache est une reconstruction de courts extraits IMDb, un score plus eleve signifie que la sortie reconstruite ressemble davantage a la reference.

## 19. Limites de cette partie

Cette partie est volontairement reduite pour rester executable sur un ordinateur portable.

Limites :

- seulement une partie du dataset IMDb est utilisee;
- le vocabulaire est limite;
- les sequences sont courtes;
- le Seq2Seq reconstruit des extraits au lieu de traduire;
- les modeles sont petits pour reduire le temps d'entrainement.

Malgre ces limites, la partie couvre les concepts principaux demandes : tokenisation, vocabulaire, RNN, LSTM, GRU, BPTT, clipping, perplexite, Seq2Seq, greedy decoding, beam search et evaluation.

## 20. Reponse de synthese - Partie III

Les architectures recurrentes permettent de modeliser des sequences car elles traitent les tokens dans l'ordre et conservent un etat cache representant le contexte precedent. Sur un corpus IMDb, un modele de langage apprend a predire le prochain token a partir des tokens precedents, ce qui correspond directement a la factorisation probabiliste d'une sequence.

Un RNN simple constitue une premiere solution, mais il peut etre instable ou insuffisant pour les dependances longues. Le passage vers LSTM ou GRU est justifie par l'utilisation de mecanismes de portes, qui ameliorent la memorisation du contexte et la stabilite de l'apprentissage. Le gradient clipping est egalement utile pour limiter les explosions de gradients pendant la retropropagation a travers le temps.

Le schema Seq2Seq devient pertinent lorsque l'objectif n'est plus seulement de predire le prochain token, mais de produire une sequence complete a partir d'une autre sequence. Dans notre cas, l'encodeur lit un extrait de critique IMDb et le decodeur apprend a reconstruire cet extrait. Le decodage glouton est simple, tandis que le beam search explore plusieurs hypotheses et peut produire des sequences plus coherentes.

Ainsi, les modeles recurrentes montrent leur interet pour le texte, mais leurs performances dependent fortement de la taille du corpus, de la longueur des sequences, du vocabulaire, du temps d'entrainement et de la strategie de decodage.

