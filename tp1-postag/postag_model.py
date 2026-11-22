#!/usr/bin/env python3
"""
Modèle RNN d'étiquetage POS, partagé par train_postag.py et predict_postag.py (Sec. 6).
"""
import os, sys
import torch
import torch.nn as nn

# Accès à lib/conllulib.py du dépôt pstal-etu (situé à côté de pstal-bmk)
LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pstal-etu", "lib")
sys.path.insert(0, os.path.abspath(LIB_DIR))

PAD_ID = 0   # indice du padding dans Vw et Vt
UNK_ID = 1   # indice des mots inconnus (OOV) dans Vw
PAD, UNK = "<PAD>", "<UNK>"

################################################################################

class GRUTagger(nn.Module):
  """
  Étiqueteur de séquences : Embedding -> GRU (bi)directionnel -> Linear.
  Contrairement à GRUClassifier (une sortie par phrase), on garde la sortie
  du GRU à *chaque* position pour prédire une étiquette par mot.
  """

  def __init__(self, d_embed, d_hidden, d_in, d_out, bidirectional=True, dropout=0.3):
    """
    d_embed : dimension des embeddings (de)
    d_hidden : dimension de l'état caché du GRU (dh)
    d_in : taille du vocabulaire des mots |Vw|
    d_out : nombre d'étiquettes |Vt|
    """
    super().__init__()
    self.embed = nn.Embedding(d_in, d_embed, padding_idx=PAD_ID)
    self.gru = nn.GRU(d_embed, d_hidden, batch_first=True, bias=False,
                      bidirectional=bidirectional)
    self.dropout = nn.Dropout(dropout)
    d_gru_out = 2 * d_hidden if bidirectional else d_hidden
    self.decision = nn.Linear(d_gru_out, d_out)

  def forward(self, idx_words):
    """
    idx_words : LongTensor B x L
    Retourne les logits : B x L x |Vt|
    """
    embedded = self.dropout(self.embed(idx_words))   # B x L x de
    hidden, _ = self.gru(embedded)                   # B x L x dh (x2 si bidir)
    return self.decision(self.dropout(hidden))       # B x L x |Vt|
