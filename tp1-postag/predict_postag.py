#!/usr/bin/env python3
"""
Prédiction des POS avec un modèle entraîné par train_postag.py (Sec. 8 du TP1).

Exemple :
  python predict_postag.py model.pt ../../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
                           -o pred.dev.conllu
"""
import argparse, sys
import torch

from postag_model import GRUTagger, UNK
from conllulib import CoNLLUReader, Util

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Prédit les POS d'un corpus CoNLL-U.")
  parser.add_argument("model", help="Modèle .pt produit par train_postag.py")
  parser.add_argument("test", help="Corpus CoNLL-U à étiqueter")
  parser.add_argument("-o", "--output", default=None, help="Fichier de sortie (défaut : stdout)")
  args = parser.parse_args()

  # 1. Charger le modèle et les vocabulaires
  saved = torch.load(args.model, weights_only=False)
  hp, wordvocab, tagvocab = saved["hyperparams"], saved["wordvocab"], saved["tagvocab"]
  model = GRUTagger(hp["d_embed"], hp["d_hidden"], len(wordvocab), len(tagvocab),
                    hp["bidirectional"], hp["dropout"])
  model.load_state_dict(saved["model_params"])
  model.eval()
  rev_tagvocab = Util.rev_vocab(tagvocab)

  # 2. Prédire phrase par phrase (pas de troncage ni de padding)
  out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
  with open(args.test, encoding="utf-8") as f:
    reader = CoNLLUReader(f)
    first = True
    for sent in reader.readConllu():
      ids = CoNLLUReader.to_int_from_vocab_sent(sent, ["form"], UNK, {"form": wordvocab})["form"]
      with torch.no_grad():
        logits = model(torch.LongTensor([ids]))          # 1 x n x |Vt|
      t_hat = logits.argmax(dim=-1)[0].tolist()
      for tok, t in zip(sent, t_hat):
        tok["upos"] = rev_tagvocab[t]
      # On recopie l'en-tête global.columns (meta-donnée de la 1re phrase)
      if first and "global.columns" not in sent.metadata:
        out.write("# global.columns = " + " ".join(c.upper() for c in reader.header) + "\n")
      first = False
      out.write(sent.serialize())
  if args.output:
    out.close()
