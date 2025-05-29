from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
import os

app = Flask(__name__)

# ==========================
# 🚀 CONFIGURAÇÃO DO BANCO
# ==========================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///academia.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ==========================
# 🚀 MODELO DO BANCO
# ==========================
class Cliente(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(100), nullable=False)
    idade = db.Column(db.Integer, nullable=False)
    tempo = db.Column(db.Integer, nullable=False)
    frequencia = db.Column(db.Integer, nullable=False)
    plano = db.Column(db.String(50), nullable=False)
    forma_pagamento = db.Column(db.String(50), nullable=False)
    feedback = db.Column(db.Integer, nullable=False)
    objetivo = db.Column(db.String(50), nullable=False)
    gosta_treino = db.Column(db.Integer, nullable=False)
    frequenta_fds = db.Column(db.Integer, nullable=False)
    reclamacoes = db.Column(db.Integer, nullable=False)
    usa_personal = db.Column(db.Integer, nullable=False)
    tem_dores = db.Column(db.Integer, nullable=False)
    usa_suplemento = db.Column(db.Integer, nullable=False)
    previsao = db.Column(db.String(100), nullable=True)

# ==========================
# 🚀 CARREGAR MODELO IA
# ==========================
MODEL_PATH = os.path.join('modelos', 'modelo_academia.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        modelo, scaler = pickle.load(f)
    print("✅ Modelo e scaler carregados com sucesso.")
except Exception as e:
    print(f"❌ ERRO ao carregar o modelo: {e}")
    modelo = None
    scaler = None

# ==========================
# 🔥 ROTAS DE TELAS
# ==========================
@app.route('/')
def home():
    clientes = Cliente.query.all()
    return render_template('index.html', clientes=clientes)

@app.route('/graficos')
def graficos():
    clientes = Cliente.query.all()
    risco = sum(1 for c in clientes if c.previsao and '⚠️' in c.previsao)
    seguro = sum(1 for c in clientes if c.previsao and '✅' in c.previsao)
    return render_template('graficos.html', dados=clientes, risco=risco, seguro=seguro)

# ==========================
# 🔥 API — CADASTRO CLIENTE
# ==========================
@app.route('/add_cliente', methods=['POST'])
def add_cliente():
    data = request.get_json()

    cliente = Cliente(
        nome=data['nome'],
        idade=data['idade'],
        tempo=data['tempo'],
        frequencia=data['frequencia'],
        plano=data['plano'],
        forma_pagamento=data['forma_pagamento'],
        feedback=data['feedback'],
        objetivo=data['objetivo'],
        gosta_treino=data['gosta_treino'],
        frequenta_fds=data['frequenta_fds'],
        reclamacoes=data['reclamacoes'],
        usa_personal=data['usa_personal'],
        tem_dores=data['tem_dores'],
        usa_suplemento=data['usa_suplemento'],
        previsao=""
    )
    db.session.add(cliente)
    db.session.commit()

    return jsonify({'message': 'Cliente cadastrado com sucesso'})

# ==========================
# 🔥 API — FAZER PREVISÃO
# ==========================
@app.route('/predict/<int:cliente_id>', methods=['POST'])
def predict(cliente_id):
    if not modelo or not scaler:
        return jsonify({'erro': 'Modelo não carregado no servidor.'}), 500

    cliente = Cliente.query.get(cliente_id)

    if not cliente:
        return jsonify({'erro': 'Cliente não encontrado'}), 404

    try:
        features = np.array([[
            cliente.idade, cliente.tempo, cliente.frequencia,
            1 if cliente.plano.lower() == 'premium' else 0,
            1 if cliente.forma_pagamento.lower() == 'trimestral' else (2 if cliente.forma_pagamento.lower() == 'anual' else 0),
            cliente.feedback,
            {'emagrecimento': 0, 'hipertrofia': 1, 'resistência': 2, 'saúde': 3}.get(cliente.objetivo.lower(), 0),
            cliente.gosta_treino,
            cliente.frequenta_fds,
            cliente.reclamacoes,
            cliente.usa_personal,
            cliente.tem_dores,
            cliente.usa_suplemento
        ]])

        features_scaled = scaler.transform(features)
        resultado = modelo.predict(features_scaled)[0]
        probabilidade = modelo.predict_proba(features_scaled)[0]
        prob_cancelamento = round(probabilidade[1] * 100, 2)

        if resultado == 1:
            texto_previsao = f"⚠️ {prob_cancelamento}% risco"
        else:
            texto_previsao = f"✅ Seguro ({100 - prob_cancelamento}%)"

        cliente.previsao = texto_previsao
        db.session.commit()

        return jsonify({
            'previsao': texto_previsao,
            'cancelamento_previsto': int(resultado),
            'probabilidade_cancelamento': prob_cancelamento
        })

    except Exception as e:
        return jsonify({'erro': f'Erro na previsão: {str(e)}'}), 500

# ==========================
# 🚀 RODAR APP
# ==========================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
