from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# Configurações de caminho
MODELO_PATH = os.path.join('modelos', 'modelo_academia.pkl')

# Inicializar o app Flask
app = Flask(__name__)

# Carregar o modelo e o scaler
try:
    with open(MODELO_PATH, 'rb') as f:
        modelo, scaler = pickle.load(f)
    print("✅ Modelo carregado com sucesso.")
except FileNotFoundError:
    print("❌ ERRO: Arquivo do modelo não encontrado.")
    exit()

@app.route('/home')
def home():
    return render_template('home.html')

# Rota principal (Página HTML)
@app.route('/')
def home():
    return render_template('index.html')

# Rota de previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if 'features' not in data:
            return jsonify({'error': 'Os dados precisam ter a chave "features".'}), 400

        entrada = np.array(data['features']).reshape(1, -1)
        entrada_scaled = scaler.transform(entrada)

        predicao = modelo.predict(entrada_scaled)
        probabilidade = modelo.predict_proba(entrada_scaled)[0][1] * 100

        resultado = {
            'cancelamento_previsto': int(predicao[0]),
            'probabilidade_cancelamento': round(probabilidade, 2)
        }

        return jsonify(resultado)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Executar o app
if __name__ == '__main__':
    app.run(debug=True)
