import os
import joblib
import numpy as np
from app import app, db, Cliente
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 🔁 Abrir contexto do Flask
with app.app_context():
    # 📥 Coletar dados do banco
    clientes = Cliente.query.all()

    X = []
    y = []

    for cliente in clientes:
        dados = [
            cliente.idade,
            cliente.tempo_treino,
            cliente.frequencia_semanal
        ]
        X.append(dados)
        y.append(cliente.cancelou)

    # ➗ Dividir os dados
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ⚙️ Escalar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 🧠 Treinar modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train_scaled, y_train)

    # 💾 Salvar modelo e scaler
    if not os.path.exists('modelos'):
        os.makedirs('modelos')

    joblib.dump(modelo, 'modelos/modelo.pkl')
    joblib.dump(scaler, 'modelos/scaler.pkl')

    print("✅ Modelo treinado e salvo com sucesso!")
