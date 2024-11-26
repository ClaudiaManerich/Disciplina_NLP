# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 23:43:49 2024

@author: Claudia Secad
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
import os

# Caminho do arquivo de dados
data_file = r"C:\Users\secad\Downloads\treino_re8\re8.csv"
if not os.path.exists(data_file):
    raise FileNotFoundError(f"O arquivo {data_file} não foi encontrado.")

# Carregando a base de dados
data = pd.read_csv(data_file)

# Verificando a existência da variável de saída
if "class" not in data.columns:
    raise ValueError("A coluna 'class' (variável de saída) não foi encontrada no arquivo CSV.")

# Separando as features e a variável alvo
X = data.drop(columns=["class"])  # Removendo a variável alvo
y = data["class"]

# Convertendo variáveis categóricas para numéricas, se necessário
X = pd.get_dummies(X, drop_first=True)
y = y.astype("category").cat.codes  # Converte a variável alvo para números

# Dividindo em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Salvando as bases de treino e teste para Avaliação Final
X_train.to_csv("train_features.csv", index=False)
y_train.to_csv("train_labels.csv", index=False)
X_test.to_csv("test_features.csv", index=False)
y_test.to_csv("test_labels.csv", index=False)

# Configuração de modelos e hiperparâmetros para busca
models_and_params = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
    },
    "SVM": {
        "model": SVC(random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
    },
}

# Realizando a busca pelos melhores hiperparâmetros (Greedy Search)
results = []
for model_name, model_info in models_and_params.items():
    print(f"Buscando hiperparâmetros para {model_name}...")
    grid = GridSearchCV(
        model_info["model"],
        model_info["params"],
        cv=5,
        scoring=make_scorer(accuracy_score),
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    # Salvando os resultados
    for params, mean_score in zip(grid.cv_results_["params"], grid.cv_results_["mean_test_score"]):
        results.append({
            "Model": model_name,
            "Params": params,
            "CV_Score": mean_score,
        })

# Convertendo os resultados em um DataFrame
results_df = pd.DataFrame(results)

# Salvando os resultados para um arquivo .csv
results_df.to_csv("hyperparameter_search_results.csv", index=False)
print("Resultados salvos em 'hyperparameter_search_results.csv'.")
