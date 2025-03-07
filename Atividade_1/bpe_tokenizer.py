# -*- coding: utf-8 -*-
"""BPE_Tokenizer

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18D_XNxgsP-F9WHI5fxnltxjjLtuqh2MU
"""

from collections import defaultdict

class BPE_Tokenizer:

    def __init__(self):
        self.vocab = {}
        self.merges = {}

    def get_stats(self, ids):
        """Calcula a frequência de pares consecutivos."""
        pairs = defaultdict(int)
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            pairs[pair] += 1
        return pairs

    def merge(self, ids, pair, idx):
        """Substitui todas as ocorrências de um par por um novo índice."""
        merged_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                merged_ids.append(idx)
                i += 2  # Pula o par já substituído
            else:
                merged_ids.append(ids[i])
                i += 1
        return merged_ids

    def train(self, text, vocab_size):
        """Treina o vocabulário usando Byte Pair Encoding (BPE)."""
        assert vocab_size >= 276, "O tamanho do vocabulário deve ser >= 276."

        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                print("Nenhum par encontrado. Encerrando o treinamento.")
                break  # Para se não houver mais pares para combinar

            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        """Decodifica uma lista de IDs em uma string."""
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode(self, text):
        """Codifica uma string em IDs usando o vocabulário treinado."""
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")), default=None)
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)

        return ids