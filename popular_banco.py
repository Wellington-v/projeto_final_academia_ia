import random
from faker import Faker
from app import app, db, Cliente

fake = Faker('pt_BR')
planos = ['Mensal', 'Trimestral', 'Anual']

with app.app_context():
    # Limpa e recria o banco
    db.drop_all()
    db.create_all()

    for _ in range(50):
        nome = fake.name()
        idade = random.randint(18, 60)
        sexo = random.choice(['M', 'F'])
        tempo_treino = round(random.uniform(0.5, 36.0), 1)
        frequencia = random.randint(1, 6)
        plano = random.choice(planos)
        cancelou = random.choice([0, 1])

        cliente = Cliente(
            nome=nome,
            idade=idade,
            sexo=sexo,
            tempo_treino=tempo_treino,
            frequencia_semanal=frequencia,
            plano=plano,
            cancelou=cancelou,
            previsao=''
        )
        db.session.add(cliente)

    db.session.commit()
    print("🌟 Banco de dados preenchido com 50 clientes fictícios!")
