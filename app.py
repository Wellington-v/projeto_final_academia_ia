from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# 🚀 Caminhos dos arquivos
MODEL_PATH = os.path.join('modelos', 'modelo_academia.pkl')
DATA_PATH = os.path.join('dados', 'dados_academia.csv')

# 🚀 Carregar o modelo e o scaler
try:
    with open(MODEL_PATH, 'rb') as f:
        modelo, scaler = pickle.load(f)
    print("✅ Modelo e scaler carregados com sucesso.")
except FileNotFoundError:
    print("❌ ERRO: Arquivo do modelo não encontrado. Verifique o caminho e o nome.")
    exit()

# ===================================
# 🔥 ROTAS DO FRONTEND (TELAS)
# ===================================

# 🔹 Home
@app.route('/')
def home():
    return render_template('home.html')

# 🔹 Explicação (Sobre a IA)
@app.route('/explicacao')
def explicacao():
    return render_template('explicacao.html')

# 🔹 Página de Contato
@app.route('/contato')
def contato():
    return render_template('contato.html')

# 🔹 Página de Gráficos
@app.route('/graficos')
def graficos():
    try:
        dados = pd.read_csv(DATA_PATH)

        risco = dados[dados['Status'] == 'Cancelado'].shape[0]
        seguro = dados[dados['Status'] == 'Ativo'].shape[0]

        dados_vis = dados[['Idade', 'Sexo', 'Tempo_meses', 'Frequencia_semanal', 'Status']].values.tolist()

        return render_template(
            'graficos.html',
            dados=dados_vis,
            risco=risco,
            seguro=seguro
        )

    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return f"Erro ao carregar dados: {e}"

# 🔹 Página de Previsão (Interface)
@app.route('/previsao')
def previsao():
    return render_template('index.html')

# ===================================
# 🔥 API DE PREVISÃO (BACKEND)
# ===================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'features' not in data:
            return jsonify({'erro': 'Dados de entrada inválidos. Chave "features" não encontrada.'}), 400

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
        print(f"❌ Erro na previsão: {e}")
        return jsonify({'erro': str(e)}), 500

# ===================================
# 🚀 RODAR O APP
# ===================================
if __name__ == '__main__':
    app.run(debug=True)
