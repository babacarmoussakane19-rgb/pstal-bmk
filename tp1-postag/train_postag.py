#!/usr/bin/env python3
"""
Entraînement d'un étiqueteur POS à base de GRU (Sec. 5 et 7 du TP1).

Exemple :
  python train_postag.py ../../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.train \
                         ../../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.dev \
                         -o model.pt
"""
import argparse
import torch
import torch.nn as nn

from postag_model import GRUTagger, PAD_ID, UNK_ID, PAD, UNK
from conllulib import CoNLLUReader, Util

################################################################################

def read_train(filename):
  """Lit le corpus d'entraînement : renvoie les phrases encodées + vocabulaires Vw, Vt."""
  with open(filename, encoding="utf-8") as f:
    int_list, vocab = CoNLLUReader(f).to_int_and_vocab({"form": [PAD, UNK], "upos": [PAD]})
  assert vocab["form"][PAD] == PAD_ID and vocab["form"][UNK] == UNK_ID
  assert vocab["upos"][PAD] == PAD_ID
  return int_list["form"], int_list["upos"], vocab

def read_dev(filename, vocab):
  """Lit le corpus dev avec les vocabulaires du train (mots inconnus -> UNK_ID)."""
  with open(filename, encoding="utf-8") as f:
    int_list = CoNLLUReader(f).to_int_from_vocab(["form", "upos"], UNK, vocab)
  return int_list["form"], int_list["upos"]

################################################################################

def pad_tensor(seqs, max_len):
  """Tronque à max_len et complète avec PAD_ID : LongTensor len(seqs) x max_len."""
  res = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
  for i, seq in enumerate(seqs):
    n = min(max_len, len(seq))
    res[i, :n] = torch.LongTensor(seq[:n])
  return res

def create_dataloader(words, tags, max_len, batch_size, shuffle):
  return Util.dataloader([pad_tensor(words, max_len)], [pad_tensor(tags, max_len)],
                         batch_size=batch_size, shuffle=shuffle)

################################################################################

def perf(model, loader, criterion):
  """Loss moyenne et accuracy (sans le padding) sur un DataLoader."""
  model.eval()
  total_loss, correct, total = 0.0, 0, 0
  with torch.no_grad():
    for X, y in loader:
      y_scores = model(X)                                   # B x L x |Vt|
      total_loss += criterion(y_scores.transpose(1, 2), y).item()
      t_hat = y_scores.argmax(dim=-1)                       # B x L
      mask = (y != PAD_ID)
      correct += ((t_hat == y) * mask).sum().item()
      total += mask.sum().item()
  return total_loss / len(loader), correct / total

def fit(model, train_loader, dev_loader, epochs, lr):
  # ignore_index : les positions de padding ne contribuent pas à la loss
  criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for X, y in train_loader:
      optimizer.zero_grad()
      y_scores = model(X)
      # CrossEntropyLoss attend B x C x L -> on transpose les 2 dernières dims
      loss = criterion(y_scores.transpose(1, 2), y)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    dev_loss, dev_acc = perf(model, dev_loader, criterion)
    print(f"epoch {epoch+1:2d}  train_loss={total_loss/len(train_loader):.4f}  "
          f"dev_loss={dev_loss:.4f}  dev_acc={dev_acc*100:.2f}%", flush=True)

################################################################################

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Entraîne un étiqueteur POS GRU.",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("train", help="Corpus d'entraînement CoNLL-U")
  parser.add_argument("dev", help="Corpus de développement CoNLL-U")
  parser.add_argument("-o", "--output", default="model.pt", help="Fichier du modèle sauvegardé")
  parser.add_argument("--d-embed", type=int, default=150, help="Dimension des embeddings (de)")
  parser.add_argument("--d-hidden", type=int, default=150, help="Dimension cachée du GRU (dh)")
  parser.add_argument("--no-bidir", action="store_true", help="GRU unidirectionnel")
  parser.add_argument("--dropout", type=float, default=0.3)
  parser.add_argument("--max-len", type=int, default=40, help="Longueur max L (troncage/padding)")
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--lr", type=float, default=0.001)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  Util.init_seed(args.seed)

  train_words, train_tags, vocab = read_train(args.train)
  dev_words, dev_tags = read_dev(args.dev, vocab)
  print(f"|Vw|={len(vocab['form'])}  |Vt|={len(vocab['upos'])}  "
        f"train={len(train_words)} phrases  dev={len(dev_words)} phrases")

  train_loader = create_dataloader(train_words, train_tags, args.max_len, args.batch_size, True)
  dev_loader = create_dataloader(dev_words, dev_tags, args.max_len, args.batch_size, False)

  hp = {"d_embed": args.d_embed, "d_hidden": args.d_hidden, "bidirectional": not args.no_bidir,
        "dropout": args.dropout, "PAD_ID": PAD_ID, "UNK_ID": UNK_ID}
  model = GRUTagger(hp["d_embed"], hp["d_hidden"], len(vocab["form"]), len(vocab["upos"]),
                    hp["bidirectional"], hp["dropout"])
  print(f"Paramètres entraînables : {Util.count_params(model)}")

  fit(model, train_loader, dev_loader, args.epochs, args.lr)

  torch.save({"wordvocab": dict(vocab["form"]), "tagvocab": dict(vocab["upos"]),
              "model_params": model.state_dict(), "hyperparams": hp}, args.output)
  print(f"Modèle sauvegardé dans {args.output}")
