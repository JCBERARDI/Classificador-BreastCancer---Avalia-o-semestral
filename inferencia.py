"""
Módulo de Inferência - Base UCI Breast Cancer (ID 14 - Ljubljana)
Carrega o modelo de melhor acurácia treinado e determina o indicativo de
diagnóstico (atributo 'Class': no-recurrence-events ou recurrence-events)
para uma nova paciente.
"""

import os
import pickle
import pandas as pd

MODELO_SALVO = "modelo_breast_cancer.pkl"

# Domínios válidos segundo a documentação oficial da UCI (ID 14)
DOMINIOS_VALIDOS = {
    "age": ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
    "menopause": ["lt40", "ge40", "premeno"],
    "tumor-size": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59"],
    "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"],
    "node-caps": ["yes", "no"],
    "deg-malig": [1, 2, 3],
    "breast": ["left", "right"],
    "breast-quad": ["left_up", "left_low", "right_up", "right_low", "central"],
    "irradiat": ["yes", "no"]
}


def carregar_modelo(caminho: str = MODELO_SALVO) -> dict:
    """Carrega o pipeline e o codificador de classes salvos pelo treinamento."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Arquivo '{caminho}' não encontrado. Execute 'python treinamento.py' primeiro."
        )
    with open(caminho, "rb") as f:
        return pickle.load(f)


def diagnosticar_paciente(dados_paciente: dict, artefato: dict = None) -> dict:
    """
    Recebe um dicionário com os 9 atributos clínicos da nova paciente e retorna
    a classe predita ('no-recurrence-events' ou 'recurrence-events') e probabilidades.
    """
    if artefato is None:
        artefato = carregar_modelo()

    pipeline = artefato["pipeline"]
    le_alvo = artefato["label_encoder"]
    colunas = artefato["colunas"]

    # Monta DataFrame na ordem exata esperada pelo ColumnTransformer
    df_paciente = pd.DataFrame([dados_paciente])[colunas]

    # Predição da classe e probabilidade
    pred_idx = pipeline.predict(df_paciente)[0]
    classe_predita = le_alvo.inverse_transform([pred_idx])[0]

    probabilidades = {}
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_paciente)[0]
        for idx, classe_nome in enumerate(le_alvo.classes_):
            probabilidades[classe_nome] = round(float(probs[idx]) * 100, 2)

    return {
        "Class": classe_predita,
        "probabilidades_pct": probabilidades,
        "modelo_utilizado": artefato["modelo_nome"]
    }


def entrada_interativa() -> dict:
    """Solicita os 9 atributos clínicos da nova paciente via terminal."""
    print("\nInforme os dados clínicos da nova paciente (pressione Enter para usar o padrão):")
    exemplos_padrao = {
        "age": "40-49",
        "menopause": "premeno",
        "tumor-size": "25-29",
        "inv-nodes": "0-2",
        "node-caps": "no",
        "deg-malig": 2,
        "breast": "left",
        "breast-quad": "left_up",
        "irradiat": "no"
    }

    paciente = {}
    for atributo, opcoes in DOMINIOS_VALIDOS.items():
        padrao = exemplos_padrao[atributo]
        opcoes_str = ", ".join(map(str, opcoes))
        while True:
            valor = input(f"  - {atributo} [{opcoes_str}] (padrão: {padrao}): ").strip()
            if valor == "":
                paciente[atributo] = padrao
                break
            if atributo == "deg-malig":
                if valor in ["1", "2", "3"]:
                    paciente[atributo] = int(valor)
                    break
            elif valor in opcoes:
                paciente[atributo] = valor
                break
            print(f"    [!] Valor inválido. Escolha entre: {opcoes_str}")

    return paciente


def main():
    print("=" * 75)
    print("MÓDULO DE INFERÊNCIA - INDICATIVO DE DIAGNÓSTICO (ATRIBUTO CLASS)")
    print("=" * 75)

    artefato = carregar_modelo()
    print(f"Modelo carregado: {artefato['modelo_nome']} (Acurácia validação: {artefato['acuracia_teste']*100:.2f}%)")

    # Demonstração com 2 perfis clínicos distintos antes da entrada manual
    pacientes_teste = [
        {
            "descricao": "Paciente Perfil A (Tumor menor, grau 1, sem envolvimento nodal)",
            "dados": {
                "age": "50-59", "menopause": "ge40", "tumor-size": "15-19",
                "inv-nodes": "0-2", "node-caps": "no", "deg-malig": 1,
                "breast": "right", "breast-quad": "left_up", "irradiat": "no"
            }
        },
        {
            "descricao": "Paciente Perfil B (Tumor maior, grau 3, cápsula nodal comprometida)",
            "dados": {
                "age": "40-49", "menopause": "premeno", "tumor-size": "35-39",
                "inv-nodes": "6-8", "node-caps": "yes", "deg-malig": 3,
                "breast": "left", "breast-quad": "left_low", "irradiat": "yes"
            }
        }
    ]

    print("\n--- TESTE AUTOMÁTICO DE INFERÊNCIA ---")
    for item in pacientes_teste:
        resultado = diagnosticar_paciente(item["dados"], artefato)
        print(f"\n{item['descricao']}:")
        print(f"  -> Diagnóstico Indicativo (Class): {resultado['Class']}")
        if resultado["probabilidades_pct"]:
            print(f"  -> Probabilidades: {resultado['probabilidades_pct']}")

    # Entrada manual para nova paciente
    print("\n--- CONSULTA PARA NOVA PACIENTE ---")
    opcao = input("Deseja inserir os dados de uma nova paciente manualmente? (s/n): ").strip().lower()
    if opcao == "s":
        nova_paciente = entrada_interativa()
        resultado = diagnosticar_paciente(nova_paciente, artefato)
        print("\n" + "-" * 55)
        print(f"INDICATIVO DE DIAGNÓSTICO (Class): {resultado['Class']}")
        if resultado["probabilidades_pct"]:
            print(f"Probabilidades estimadas: {resultado['probabilidades_pct']}")
        print("-" * 55)


if __name__ == "__main__":
    main()