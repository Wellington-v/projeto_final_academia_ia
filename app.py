from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Caminhos dos arquivos
MODEL_PATH = os.path.join('modelos', 'modelo_academia.pkl')
DATA_PATH = os.path.join('dados', 'dados_academia.csv')

# Carregar o modelo e o scaler
try:
    with open(MODEL_PATH, 'rb') as f:
        modelo, scaler = pickle.load(f)
    print("✅ Modelo carregado com sucesso.")
except FileNotFoundError:
    print("❌ ERRO: Arquivo do modelo não encontrado.")
    exit()

# 🔥 Rota HOME (Boas-vindas)
@app.route('/')
def home():
    return render_template('home.html')

# 🔥 Rota EXPLICAÇÃO (Sobre a IA)
@app.route('/explicacao')
def explicacao():
    return render_template('explicacao.html')

@app.route('/contato')
def contato():
    return render_template('contato.html')


# 🔥 Rota GRÁFICOS
@app.route('/graficos')
def graficos():
    try:
        dados = pd.read_csv(DATA_PATH)

        risco = dados[dados['Status'] == 'Cancelado'].shape[0]
        seguro = dados[dados['Status'] == 'Ativo'].shape[0]

        dados_vis = dados[['Idade', 'Sexo', 'Tempo_meses', 'Frequencia_semanal', 'Status']].values.tolist()

        return render_template('graficos.html', dados=dados_vis, risco=risco, seguro=seguro)

    except Exception as e:
        return f"Erro ao carregar dados: {e}"

# 🔥 Rota da PÁGINA DE PREVISÃO
@app.route('/previsao')
def previsao():
    return render_template('index.html')

# 🔥 API de previsão (backend)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)

        resultado = modelo.predict(features_scaled)[0]
        probabilidade = modelo.predict_proba(features_scaled)[0]

        prob_cancelamento = round(probabilidade[1] * 100, 2)
        retorno = {
            'cancelamento_previsto': int(resultado),
            'probabilidade_cancelamento': prob_cancelamento
        }

        return jsonify(retorno)

    except Exception as e:
        return jsonify({'erro': str(e)})

# 🔥 Rodar o app
if __name__ == '__main__':
    app.run(debug=True)
