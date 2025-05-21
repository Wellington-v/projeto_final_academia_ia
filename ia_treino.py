import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# Configurações de caminho
DADOS_PATH = os.path.join('dados', 'dados_academia.csv')
MODELO_PATH = os.path.join('modelos', 'modelo_academia.pkl')

# Função para carregar dados
def carregar_dados(caminho):
    try:
        dados = pd.read_csv(caminho)
        print(f"✅ Dados carregados ({dados.shape[0]} registros, {dados.shape[1]} colunas)")
        return dados
    except FileNotFoundError:
        print("❌ Erro: Arquivo de dados não encontrado.")
        exit()

# Função de pré-processamento
def preprocessar_dados(dados):
    dados = dados.dropna().drop_duplicates()

    le = LabelEncoder()
    for coluna in dados.select_dtypes(include=['object']).columns:
        dados[coluna] = le.fit_transform(dados[coluna])

    X = dados.drop('cancelado', axis=1)
    y = dados['cancelado']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Execução
if __name__ == "__main__":
    print("🔧 Iniciando treinamento do modelo...")

    dados = carregar_dados(DADOS_PATH)
    X, y, scaler = preprocessar_dados(dados)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n📊 Avaliação do modelo:")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    os.makedirs('modelos', exist_ok=True)
    with open(MODELO_PATH, 'wb') as f:
        pickle.dump((modelo, scaler), f)

    print("\n✅ Modelo treinado e salvo com sucesso!")
