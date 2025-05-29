import os
import sqlite3
import random

# Criar a pasta do banco se não existir
os.makedirs('database', exist_ok=True)

# Conectar ao banco
conn = sqlite3.connect('database/clientes.db')
cursor = conn.cursor()

# Criar a tabela se não existir
cursor.execute("""
    CREATE TABLE IF NOT EXISTS clientes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT,
        idade INTEGER,
        sexo TEXT,
        frequencia_semanal INTEGER,
        tempo_matricula INTEGER,
        status TEXT
    );
""")

# Listas para nomes
nomes_femininos = ['Ana', 'Beatriz', 'Camila', 'Daniela', 'Eduarda', 'Fernanda', 'Gabriela', 'Helena', 'Isabela', 'Joana']
nomes_masculinos = ['Carlos', 'Diego', 'Eduardo', 'Felipe', 'Gustavo', 'Henrique', 'Igor', 'João', 'Lucas', 'Matheus']
sobrenomes = ['Silva', 'Souza', 'Oliveira', 'Pereira', 'Costa', 'Almeida', 'Rocha', 'Martins', 'Melo', 'Fernandes']

# Gerar 50 clientes falsos
for _ in range(50):
    sexo = random.choice(['Feminino', 'Masculino'])
    nome = random.choice(nomes_femininos if sexo == 'Feminino' else nomes_masculinos) + ' ' + random.choice(sobrenomes)
    idade = random.randint(18, 50)
    frequencia = random.randint(1, 5)
    tempo_matricula = random.randint(1, 24)
    status = random.choice(['Ativo', 'Cancelado'])

    cursor.execute("""
        INSERT INTO clientes (nome, idade, sexo, frequencia_semanal, tempo_matricula, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (nome, idade, sexo, frequencia, tempo_matricula, status))

# Salvar e fechar
conn.commit()
conn.close()

print("🌟 Banco de dados preenchido com 50 clientes fictícios!")
