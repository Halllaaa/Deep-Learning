# Annexe Experimentale

Cette annexe recense les fichiers generes pendant le projet : scripts, resultats, figures et modeles sauvegardes.

---

# 1. Scripts principaux

| Partie | Script | Role |
|---|---|---|
| Partie I | `src/part1_mlp_tabular.py` | MLP sur Breast Cancer Wisconsin |
| Partie II | `src/part2_cnn_images.py` | CNN sur Fashion-MNIST |
| Partie III | `src/part3_imdb_sequences.py` | RNN/LSTM/GRU/Seq2Seq sur IMDb |

Commandes d'execution :

```powershell
cd "C:\Users\halas\OneDrive\Bureau\DL"
python .\src\part1_mlp_tabular.py
python .\src\part2_cnn_images.py
python -u .\src\part3_imdb_sequences.py
```

---

# 2. Resultats tabulaires

| Partie | Fichier | Contenu |
|---|---|---|
| Partie I | `outputs/tables/part1_mlp_results.csv` | Comparaison des MLP et initialisations |
| Partie II | `outputs/tables/part2_cnn_results.csv` | Comparaison MLP image et CNN |
| Partie II | `outputs/tables/part2_manual_operations.txt` | Correlation 2D et pooling manuels compares a PyTorch |
| Partie III | `outputs/tables/part3_lm_results.csv` | Resultats RNN/LSTM/GRU avec perplexite |
| Partie III | `outputs/tables/part3_seq2seq_results.csv` | Scores Seq2Seq greedy et beam search |
| Partie III | `outputs/tables/part3_seq2seq_predictions.txt` | Exemples de decodage Seq2Seq |
| Partie III | `outputs/tables/part3_language_generations.txt` | Exemples de generation de langage |

## Interpretation des tableaux

Le tableau `part1_mlp_results.csv` sert a comparer les deux implementations du MLP et les trois strategies d'initialisation. Il montre que les initialisations gaussienne et Xavier donnent les meilleurs resultats, tandis que l'initialisation constante est legerement moins performante. Ce tableau justifie donc l'importance de l'initialisation dans l'apprentissage d'un reseau de neurones.

Le tableau `part2_cnn_results.csv` sert a comparer un MLP applique a des images aplaties avec plusieurs variantes de CNN. Il montre que le CNN strided obtient le meilleur F1 macro. Ce tableau soutient l'idee que les convolutions sont plus adaptees aux images, tout en montrant que les choix architecturaux doivent etre testes experimentalement.

Le fichier `part2_manual_operations.txt` verifie que les implementations manuelles de la correlation croisee 2D, du max-pooling et de l'average-pooling donnent les memes resultats que les fonctions PyTorch correspondantes. Il relie donc les calculs theoriques aux operations utilisees dans un CNN reel.

Le tableau `part3_lm_results.csv` compare les modeles de langage RNN, LSTM et GRU avec la perplexite. Il montre que, dans cette experience courte, le RNN avec clipping obtient la meilleure perplexite. Ce resultat doit etre interprete en tenant compte du protocole reduit.

Le tableau `part3_seq2seq_results.csv` compare le decodage glouton et le beam search sur la tache Seq2Seq. Les scores faibles indiquent que la reconstruction libre de texte reste difficile avec un modele simple et peu de donnees.

---

# 3. Figures principales

## Partie I

Les figures suivantes sont generees pour chaque variante MLP :

- courbes de loss et accuracy de validation;
- matrice de confusion.

Dossier :

```text
outputs/figures
```

Exemples de fichiers :

```text
sequential_gaussian_training_curves.png
sequential_gaussian_confusion_matrix.png
custom_gaussian_training_curves.png
custom_gaussian_confusion_matrix.png
```

Interpretation attendue :

Les courbes d'apprentissage permettent de verifier la convergence du MLP. Une baisse de la loss indique que le modele apprend a reduire son erreur. L'accuracy de validation indique si cet apprentissage se generalise a des donnees non vues pendant l'entrainement.

Les matrices de confusion montrent les erreurs par classe. Pour Breast Cancer Wisconsin, elles permettent d'analyser les confusions entre tumeur maligne et tumeur benigne, ce qui est essentiel pour interpreter les resultats au-dela de l'accuracy globale.

## Partie II

Figures generees :

- courbes d'entrainement;
- matrices de confusion;
- cartes de caracteristiques CNN.

Exemples :

```text
part2_cnn_strided_training_curves.png
part2_cnn_strided_confusion_matrix.png
part2_cnn_strided_feature_maps.png
```

Interpretation attendue :

Les courbes d'entrainement montrent si les CNN apprennent progressivement des representations utiles. Les matrices de confusion permettent d'identifier les classes Fashion-MNIST les plus souvent confondues, notamment les vetements visuellement proches comme Shirt, Coat et Pullover.

Les cartes de caracteristiques montrent les activations internes du CNN. Elles illustrent le fait que les couches convolutionnelles transforment l'image en representations visuelles plus abstraites, ce qui correspond a l'objectif theorique des CNN.

## Partie III

Figures generees :

```text
part3_language_model_curves.png
part3_seq2seq_training_curves.png
```

La premiere montre la loss de validation et la perplexite des modeles de langage. La seconde montre l'evolution de la loss du modele Seq2Seq.

Interpretation attendue :

La figure des modeles de langage permet de comparer la capacite des RNN, LSTM et GRU a predire le prochain token. Une perplexite plus faible indique une meilleure modelisation de la sequence.

La figure Seq2Seq montre que la loss diminue pendant l'entrainement. Cependant, les exemples de generation montrent que la diminution de la loss ne garantit pas une bonne generation libre. Cette difference doit etre discutee dans l'analyse critique.

---

# 4. Modeles sauvegardes

Les modeles sont sauvegardes dans :

```text
outputs/models
```

## Partie I

Modeles MLP :

```text
sequential_gaussian.pt
sequential_constant.pt
sequential_xavier.pt
custom_gaussian.pt
custom_constant.pt
custom_xavier.pt
```

## Partie II

Modeles image :

```text
part2_mlp_flattened_images.pt
part2_cnn_lenet_max_padding.pt
part2_cnn_avg_pooling.pt
part2_cnn_more_filters.pt
part2_cnn_with_1x1.pt
part2_cnn_strided.pt
```

## Partie III

Modeles texte :

```text
part3_rnn_no_clipping.pt
part3_rnn_clipped.pt
part3_lstm_clipped.pt
part3_gru_clipped.pt
part3_seq2seq_gru.pt
```

---

# 5. Resume numerique des meilleurs resultats

## Partie I

Meilleurs modeles :

```text
Custom MLP + initialisation gaussienne
Sequential MLP + initialisation Xavier
```

Resultat :

```text
accuracy test = 0.9419
F1-score = 0.9524
```

## Partie II

Meilleur modele :

```text
CNN strided
```

Resultat :

```text
accuracy test = 0.8190
F1 macro = 0.8220
```

## Partie III

Meilleur modele de langage :

```text
RNN avec gradient clipping
```

Resultat :

```text
test perplexity = 1397.93
```

Resultat Seq2Seq :

```text
greedy BLEU-like score = 0.0972
beam search BLEU-like score = 0.0960
```

---

# 6. Remarques sur la reproductibilite

Les scripts fixent des graines aleatoires avec :

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

Cela rend les resultats plus stables, mais de petites differences peuvent apparaitre selon :

- la version de PyTorch;
- le CPU ou GPU utilise;
- l'ordre exact des operations;
- les bibliotheques installees.

---

# 7. Remarques sur les limites experimentales

Les experiences sont concues pour etre executables sur ordinateur portable. Les sous-ensembles reduits permettent de gagner du temps, mais limitent les performances finales.

Pour ameliorer les resultats, il serait possible de :

- utiliser le dataset complet;
- augmenter le nombre d'epochs;
- tester plus d'hyperparametres;
- ajouter early stopping;
- ajouter attention pour Seq2Seq;
- comparer avec des architectures modernes comme Transformer.
