import os
import json
import torch
import tiktoken


class BigramModel:
    # Inicializa o codificador Byte Pair Encoding (BPE) GPT-4
    __encoding = tiktoken.get_encoding("o200k_base")

    @staticmethod
    def encode(text: str) -> list[int]:
        """Codifica o texto em tokens."""
        return BigramModel.__encoding.encode(text, allowed_special={"<|endoftext|>"})

    @staticmethod
    def decode(token_list: list[int]) -> str:
        """Decodifica uma lista de tokens para texto."""
        return BigramModel.__encoding.decode(token_list)

    @staticmethod
    def generate_token_dicts(token_list: list) -> tuple[dict, dict]:
        """Cria dicionários de mapeamento entre tokens e índices."""
        tokens = set(token_list)
        tokens.add(BigramModel.encode("<|endoftext|>")[0])  # Adiciona token especial de fim de texto
        token_to_idx = {val: idx for idx, val in enumerate(tokens)}
        idx_to_token = {idx: val for val, idx in token_to_idx.items()}
        return token_to_idx, idx_to_token

    @staticmethod
    def generate_probabilities(token_list: list, token_to_index: dict,
                               debug: bool = False, fill_val: int = 0) -> torch.tensor:
        """
        Gera a matriz de probabilidades baseada na contagem de pares de tokens.
        """
        vocab_size = len(token_to_index)
        counter_tensor = torch.full((vocab_size, vocab_size), fill_val, dtype=torch.int64)

        # Itera sobre pares consecutivos para calcular frequências
        for t1, t2 in zip(token_list, token_list[1:]):
            counter_tensor[token_to_index[t1], token_to_index[t2]] += 1

        # Normaliza para obter probabilidades (divide pelo somatório da linha)
        probs = counter_tensor.to(torch.float64)
        probs /= torch.sum(probs, dim=1, keepdim=True)

        return probs

    def __init__(self, max_tokens: int = 200000):
        """Inicializa o modelo Bigram."""
        self.max_tokens = max_tokens
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.probabilities = torch.zeros([])

    def train(self, base_text: str, debug: bool = False, fill_val: int = 0):
        """
        Treina o modelo Bigram no texto fornecido.
        """
        token_list = BigramModel.encode(base_text)[:self.max_tokens]
        self.token_to_idx, self.idx_to_token = BigramModel.generate_token_dicts(token_list)
        self.probabilities = BigramModel.generate_probabilities(token_list, self.token_to_idx, debug, fill_val)

        # Garantir que todas as linhas da matriz de probabilidades sejam válidas
        for i in range(self.probabilities.shape[0]):
            if torch.sum(self.probabilities[i]) == 0:
                self.probabilities[i] = 1 / self.probabilities.shape[1]  # Distribuição uniforme

    def generate_text(self, max_tokens: int, seed: int = -1) -> str:
        """
        Gera texto com base no modelo treinado.
        """
        g = torch.Generator().manual_seed(seed if seed != -1 else torch.seed())
        token_list = []
        idx = torch.randint(0, len(self.token_to_idx), (1,), generator=g).item()
        special_idx = self.token_to_idx[BigramModel.encode("<|endoftext|>")[0]]

        for _ in range(max_tokens):
            if idx == special_idx:  # Para a geração ao atingir o token especial
                break
            token_list.append(self.idx_to_token[idx])
            idx = torch.multinomial(self.probabilities[idx], num_samples=1, replacement=True, generator=g).item()

        return BigramModel.decode(token_list)

    def perplexity(self, word: str) -> float:
        """
        Calcula a perplexidade de uma palavra baseada no modelo treinado.
        Tokens desconhecidos recebem uma probabilidade mínima.
        """
        tokens = BigramModel.encode(word)
        word_prob = []

        for t1, t2 in zip(tokens, tokens[1:]):
            if t1 in self.token_to_idx and t2 in self.token_to_idx:
                prob = self.probabilities[self.token_to_idx[t1], self.token_to_idx[t2]]
            else:
                prob = 1e-10  # Probabilidade mínima para pares desconhecidos
            word_prob.append(max(prob, 1e-10))  # Garantir que a probabilidade seja pelo menos 1e-10

        # Converte para tensor
        word_prob = torch.tensor(word_prob)
        sum_log = torch.sum(torch.log(word_prob))
        return torch.exp(-sum_log / len(word_prob)).item()

    @staticmethod
    def load_corpus(directory: str) -> str:
        """
        Carrega e combina todos os arquivos JSON de um diretório em um único texto.
        """
        combined_text = ""
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Supondo que o texto está em uma chave chamada "text"
                    combined_text += data.get("text", "") + " "
        return combined_text.strip()

    def save_model(self, path: str):
        """
        Salva o modelo treinado em um arquivo.
        """
        torch.save({
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'probabilities': self.probabilities
        }, path)

    @staticmethod
    def load_model(path: str):
        """
        Carrega um modelo Bigram previamente salvo.
        """
        checkpoint = torch.load(path)
        model = BigramModel()
        model.token_to_idx = checkpoint['token_to_idx']
        model.idx_to_token = checkpoint['idx_to_token']
        model.probabilities = checkpoint['probabilities']
        return model
