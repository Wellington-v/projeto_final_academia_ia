from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# 📦 Caminho dos arquivos
MODEL_PATH = os.path.join('modelos', 'modelo_academia.pkl')
DATA_PATH = os.path.join('dados', 'dados_academia.csv')

# ✅ Carregar modelo e scaler
try:
    with open(MODEL_PATH, 'rb') as f:
        modelo, scaler = pickle.load(f)
    print("✅ Modelo carregado com sucesso!")
except FileNotFoundError:
    print("❌ ERRO: Modelo não encontrado!")
    exit()

# ===============================
# 🔥 ROTAS DO SITE
# ===============================

# 🏠 Tela de Boas-vindas
@app.route('/')
def home():
    return render_template('home.html')


# 🔍 Tela de Previsão da IA
@app.route('/previsao')
def previsao():
    return render_template('index.html')


# 🔗 Rota para Fazer a Previsão (Backend)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        idade = float(request.form['idade'])
        sexo = float(request.form['sexo'])
        tempo_treino = float(request.form['tempo_treino'])
        frequencia = float(request.form['frequencia'])

        entrada = np.array([[idade, sexo, tempo_treino, frequencia]])
        entrada = scaler.transform(entrada)

        pred = modelo.predict(entrada)

        resultado = 'Risco de Cancelamento' if pred[0] == 1 else 'Cliente Seguro'

        return render_template('index.html', resultado=resultado)

    except Exception as e:
        print(f"Erro na previsão: {e}")
        return render_template('index.html', resultado="Erro na previsão")


# 📖 Tela de Explicação
@app.route('/explicacao')
def explicacao():
    return render_template('explicacao.html')


# 📊 Tela de Gráficos e Dashboard
@app.route('/graficos')
def graficos():
    try:
        dados = pd.read_csv(DATA_PATH)

        # Conta quantos estão em risco e quantos estão seguros
        risco = dados['cancelado'].value_counts().get(1, 0)
        seguro = dados['cancelado'].value_counts().get(0, 0)

        # Prepara os dados pra tabela
        tabela = dados.values.tolist()

        return render_template('graficos.html', dados=tabela, risco=risco, seguro=seguro)

    except Exception as e:
        print(f"Erro ao carregar gráficos: {e}")
        return render_template('graficos.html', dados=[], risco=0, seguro=0)


# ===============================
# 🚀 INICIALIZA O APP
# ===============================
if __name__ == '__main__':
    app.run(debug=True)
