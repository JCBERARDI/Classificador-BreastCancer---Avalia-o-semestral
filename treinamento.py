"""
Módulo de Treinamento - Base UCI Breast Cancer (ID 14 - Ljubljana)
Fonte: https://archive.ics.uci.edu/dataset/14/breast+cancer

Procedimentos executados:
1. Carregamento e limpeza dos dados (tratamento de valores ausentes '?')
2. Análise Exploratória (EDA) e balanceamento/distribuição das classes
3. Pré-processamento: Codificação de atributos categóricos (One-Hot Encoding / Ordinal) e normalização
4. Avaliação comparativa de modelos (Regressão Logística, Árvore de Decisão, Random Forest, KNN, SVM)
5. Seleção automática e persistência do modelo com melhor acurácia (pickle)
"""

import os
import pickle
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

URL_UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data"
ARQUIVO_LOCAL = "breast-cancer.data"
MODELO_SALVO = "modelo_breast_cancer.pkl"

COLUNAS = [
    "Class",        # Variável Alvo: no-recurrence-events, recurrence-events
    "age",          # Faixa etária: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99
    "menopause",    # Estado de menopausa: lt40, ge40, premeno
    "tumor-size",   # Tamanho do tumor (mm): 0-4, 5-9, 10-14, ..., 50-54, 55-59
    "inv-nodes",    # Nódulos linfáticos axilares envolvidos: 0-2, 3-5, ..., 36-39
    "node-caps",    # Penetração da cápsula nodal: yes, no
    "deg-malig",    # Grau histológico de malignidade: 1, 2, 3
    "breast",       # Mama afetada: left, right
    "breast-quad",  # Quadrante da mama: left_up, left_low, right_up, right_low, central
    "irradiat"      # Histórico de radioterapia: yes, no
]


def carregar_dados() -> pd.DataFrame:
    """Baixa (se necessário) e carrega a base UCI Breast Cancer."""
    if not os.path.exists(ARQUIVO_LOCAL):
        print(f"[INFO] Baixando dataset de: {URL_UCI} ...")
        urllib.request.urlretrieve(URL_UCI, ARQUIVO_LOCAL)

    df = pd.read_csv(ARQUIVO_LOCAL, names=COLUNAS, na_values="?")
    return df


def treinar_e_avaliar():
    print("=" * 75)
    print("MÓDULO DE TREINAMENTO - UCI BREAST CANCER (ID 14)")
    print("=" * 75)

    # 1. Carregamento
    df = carregar_dados()
    print(f"\n[1. CARREGAMENTO] Total de registros: {df.shape[0]} | Total de colunas: {df.shape[1]}")

    # 2. Análise Exploratória (EDA)
    print("\n[2. ANÁLISE EXPLORATÓRIA]")
    print("Valores ausentes ('?') por atributo:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string() or "  Nenhum valor ausente.")

    print("\nDistribuição da variável alvo (Class):")
    for classe, qtd in df["Class"].value_counts().items():
        pct = (qtd / len(df)) * 100
        print(f"  - {classe}: {qtd} registros ({pct:.2f}%)")

    # 3. Separação de variáveis independentes (X) e variável alvo (y)
    X = df.drop(columns=["Class"])
    y_raw = df["Class"]

    # Codificação do alvo: 0 = no-recurrence-events, 1 = recurrence-events
    le_alvo = LabelEncoder()
    y = le_alvo.fit_transform(y_raw)

    # Identificação dos tipos de atributos
    colunas_categoricas = [
        "age", "menopause", "tumor-size", "inv-nodes",
        "node-caps", "breast", "breast-quad", "irradiat"
    ]
    colunas_numericas = ["deg-malig"]

    # 4. Divisão Treino e Teste (estratificada devido ao desbalanceamento ~70/30)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\n[3. SPLIT] Treino: {X_treino.shape[0]} amostras | Teste: {X_teste.shape[0]} amostras")

    # 5. Pipeline de Pré-processamento
    # - Imputação pela moda (most_frequent) para valores ausentes ('?')
    # - OneHotEncoder para atributos nominais/ordinais em faixa
    transformador_cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformador_num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessador = ColumnTransformer(
        transformers=[
            ("cat", transformador_cat, colunas_categoricas),
            ("num", transformador_num, colunas_numericas)
        ]
    )

    # 6. Definição dos modelos candidatos para seleção da melhor acurácia
    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Árvore de Decisão": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=7),
        "Support Vector Machine (SVM)": SVC(kernel="rbf", probability=True, random_state=42)
    }

    print("\n[4. TREINAMENTO E COMPARAÇÃO DE ACURÁCIA]")
    print("-" * 75)
    melhor_nome = None
    melhor_acuracia = -1.0
    melhor_pipeline = None

    for nome, estimador in modelos.items():
        pipe = Pipeline(steps=[
            ("preprocessamento", preprocessador),
            ("classificador", estimador)
        ])

        # Validação cruzada (5-fold) no conjunto de treino
        cv_scores = cross_val_score(pipe, X_treino, y_treino, cv=5, scoring="accuracy")

        # Treino final e avaliação no conjunto de teste
        pipe.fit(X_treino, y_treino)
        y_pred = pipe.predict(X_teste)
        acc_teste = accuracy_score(y_teste, y_pred)

        print(f"{nome:<30} | CV (5-fold): {cv_scores.mean()*100:5.2f}% | Teste: {acc_teste*100:5.2f}%")

        if acc_teste > melhor_acuracia:
            melhor_acuracia = acc_teste
            melhor_nome = nome
            melhor_pipeline = pipe

    # 7. Avaliação detalhada do melhor modelo
    print("-" * 75)
    print(f"\n[5. MODELO VENCEDOR] {melhor_nome} (Acurácia no Teste: {melhor_acuracia*100:.2f}%)")
    y_pred_final = melhor_pipeline.predict(X_teste)

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_teste, y_pred_final))

    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, y_pred_final, target_names=le_alvo.classes_))

    # 8. Persistência do artefato para o módulo de inferência
    artefato = {
        "pipeline": melhor_pipeline,
        "label_encoder": le_alvo,
        "colunas": X.columns.tolist(),
        "modelo_nome": melhor_nome,
        "acuracia_teste": melhor_acuracia
    }

    with open(MODELO_SALVO, "wb") as f:
        pickle.dump(artefato, f)

    print(f"[6. PERSISTÊNCIA] Modelo e pré-processadores salvos com sucesso em '{MODELO_SALVO}'.")


if __name__ == "__main__":
    treinar_e_avaliar()