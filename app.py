from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os

# ========== CONFIGURAÇÕES ==========
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///banco.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ========== MODELO DO BANCO ==========
class Cliente(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(100), nullable=False)
    idade = db.Column(db.Integer, nullable=False)
    sexo = db.Column(db.String(1), nullable=False)
    tempo_treino = db.Column(db.Float, nullable=False)
    frequencia_semanal = db.Column(db.Integer, nullable=False)
    plano = db.Column(db.String(20), nullable=False)
    cancelou = db.Column(db.Integer, nullable=False)
    previsao = db.Column(db.String(200))

# ========== ROTAS HTML ==========
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/previsao')
def previsao():
    clientes = Cliente.query.all()
    return render_template('index.html', clientes=clientes)

@app.route('/explicacao')
def explicacao():
    return render_template('explicacao.html')

@app.route('/graficos')
def graficos():
    seguro = Cliente.query.filter(Cliente.previsao.like('%Seguro%')).count()
    risco = Cliente.query.filter(Cliente.previsao.like('%Risco%')).count()
    return render_template('graficos.html', seguro=seguro, risco=risco)

@app.route('/contato')
def contato():
    return render_template('contato.html')

# ========== ROTAS JSON/API ==========
@app.route('/clientes')
def listar_clientes():
    clientes = Cliente.query.all()
    lista = []
    for c in clientes:
        lista.append({
            'id': c.id,
            'nome': c.nome,
            'idade': c.idade,
            'sexo': c.sexo,
            'tempo_treino': c.tempo_treino,
            'frequencia_semanal': c.frequencia_semanal,
            'plano': c.plano,
            'cancelou': c.cancelou,
            'previsao': c.previsao
        })
    return jsonify(lista)

@app.route('/prever/<int:cliente_id>', methods=['POST'])
def prever_cancelamento(cliente_id):
    try:
        cliente = Cliente.query.get(cliente_id)
        if not cliente:
            return jsonify({'erro': 'Cliente não encontrado'}), 404

        modelo = joblib.load('modelos/modelo.pkl')
        scaler = joblib.load('modelos/scaler.pkl')

        dados = np.array([[cliente.idade, cliente.tempo_treino, cliente.frequencia_semanal]])
        dados_escalados = scaler.transform(dados)
        probabilidade = modelo.predict_proba(dados_escalados)[0]
        risco = round(probabilidade[1] * 100, 2)

        if modelo.predict(dados_escalados)[0] == 1:
            texto = f"⚠️ Risco de cancelamento: {risco}%"
        else:
            texto = f"✅ Seguro: {100 - risco}%"

        cliente.previsao = texto
        db.session.commit()

        return jsonify({
            'cliente_id': cliente.id,
            'nome': cliente.nome,
            'previsao': texto,
            'risco_cancelamento': risco
        })

    except Exception as e:
        return jsonify({'erro': f'Erro na previsão: {str(e)}'}), 500

@app.route('/prever_todos', methods=['POST'])
def prever_todos():
    try:
        modelo = joblib.load('modelos/modelo.pkl')
        scaler = joblib.load('modelos/scaler.pkl')

        clientes = Cliente.query.all()

        for cliente in clientes:
            dados = np.array([[cliente.idade, cliente.tempo_treino, cliente.frequencia_semanal]])
            dados_escalados = scaler.transform(dados)
            probabilidade = modelo.predict_proba(dados_escalados)[0]
            risco = round(probabilidade[1] * 100, 2)

            if modelo.predict(dados_escalados)[0] == 1:
                texto = f"⚠️ Risco de cancelamento: {risco}%"
            else:
                texto = f"✅ Seguro: {100 - risco}%"

            cliente.previsao = texto

        db.session.commit()
        return jsonify({'mensagem': 'Previsões geradas para todos os clientes!'})

    except Exception as e:
        return jsonify({'erro': f'Erro ao prever para todos: {str(e)}'}), 500

# ========== EXECUÇÃO ==========
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
