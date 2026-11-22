# pstal-bmk

Travaux pratiques du cours **PSTAL** (Prédiction de structures pour le TAL),
Master IAAA, Aix-Marseille Université.
Auteur : Babacar Moussa Kane 

Tous les TP utilisent le corpus **Sequoia** (phrases en français, format CoNLL-U).

## Organisation

```
pstal-bmk/
└── tp1-postag/          TP1 : étiquetage en parties du discours (POS)
    ├── postag_model.py      modèle GRU partagé
    ├── train_postag.py      entraînement
    ├── predict_postag.py    prédiction
    └── resultats.dev.txt    scores sur le corpus dev
```

Ce dépôt doit être placé à côté du dépôt `pstal-etu`, qui fournit les données
(`sequoia/`) et les outils (`lib/conllulib.py`, `lib/evaluate.py`).

## Prérequis

- Python 3.10
- `torch`, `conllu`

```bash
pip install torch conllu
```

## TP1 : étiqueteur POS

**Objectif** : prédire la catégorie grammaticale (colonne UPOS) de chaque mot.

**Modèle** : Embedding → GRU bidirectionnel → Linear.

### Utilisation

Depuis `tp1-postag/` :

```bash
S=../../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple

python train_postag.py $S.train $S.dev -o model.pt
python predict_postag.py model.pt $S.dev -o pred.dev.conllu
python ../../pstal-etu/lib/evaluate.py -p pred.dev.conllu -g $S.dev -t $S.train
```
