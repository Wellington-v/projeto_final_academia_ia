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
    return render_template('home.html')

@app.route('/explicacao')
def explicacao():
    return render_template('explicacao.html')

@app.route('/contato')
def contato():
    return render_template('contato.html')

@app.route('/graficos')
def graficos():
    clientes = Cliente.query.all()

    risco = sum(1 for c in clientes if c.previsao and '⚠️' in c.previsao)
    seguro = sum(1 for c in clientes if c.previsao and '✅' in c.previsao)

    return render_template(
        'graficos.html',
        dados=clientes,
        risco=risco,
        seguro=seguro
    )

@app.route('/previsao')
def previsao():
    clientes = Cliente.query.all()
    return render_template('index.html', clientes=clientes)

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
        previsao=""
    )
    db.session.add(cliente)
    db.session.commit()

    return jsonify({'message': 'Cliente cadastrado com sucesso'})

# ==========================
# 🔥 API — LISTAR CLIENTES
# ==========================
@app.route('/clientes')
def listar_clientes():
    clientes = Cliente.query.all()
    output = []
    for c in clientes:
        output.append({
            'id': c.id,
            'nome': c.nome,
            'idade': c.idade,
            'tempo': c.tempo,
            'frequencia': c.frequencia,
            'plano': c.plano,
            'previsao': c.previsao
        })
    return jsonify(output)

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
        features = np.array([[cliente.idade, cliente.tempo, cliente.frequencia, 1 if cliente.plano.lower() == 'premium' else 0]])
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
