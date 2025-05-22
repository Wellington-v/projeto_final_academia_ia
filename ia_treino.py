import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt

# Caminhos
DATA_PATH = 'dados/dados_academia.csv'
MODEL_PATH = 'modelos/modelo_academia.pkl'

# Carregar dados
dados = pd.read_csv(DATA_PATH)

# Pre-processamento
le = LabelEncoder()
for coluna in dados.select_dtypes(include=['object']).columns:
    dados[coluna] = le.fit_transform(dados[coluna])

X = dados.drop('Status', axis=1)
y = dados['Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Testar modelos
modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    resultados[nome] = acc
    print(f"\nModelo: {nome}")
    print(f"Acurácia: {acc*100:.2f}%")
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

# Escolher o melhor modelo
melhor_modelo_nome = max(resultados, key=resultados.get)
melhor_modelo = modelos[melhor_modelo_nome]

print(f"\n🔍 Melhor modelo: {melhor_modelo_nome} com acurácia de {resultados[melhor_modelo_nome]*100:.2f}%")

# Análise de importância dos dados (se suportado)
if melhor_modelo_nome in ["RandomForest", "DecisionTree", "XGBoost"]:
    importancia = melhor_modelo.feature_importances_
    plt.figure(figsize=(8,6))
    plt.barh(X.columns, importancia)
    plt.xlabel("Importância")
    plt.ylabel("Variáveis")
    plt.title(f"Importância das Variáveis — {melhor_modelo_nome}")
    plt.tight_layout()
    plt.show()

# Salvar modelo e scaler
with open(MODEL_PATH, 'wb') as f:
    pickle.dump((melhor_modelo, scaler), f)

print("\n✅ Modelo salvo com sucesso!")
