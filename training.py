# Inteligência Artificial aplicada a Negócios e Empresas - Estudo de Caso 2
# Fase de treinamento de IA

# Instalando as bibliotecas necessárias
# conda install -c conda-forge keras

# Importar bibliotecas e outros arquivos python
import os
import numpy as np
import random as rn

import environment
import brain
import dqn

# Configurar seeds para reprodutibilidade
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# CONFIGURAÇÃO DE PARÂMETROS
epsilon = 0.3
number_actions = 5
direction_boundary = (number_actions -1)/2
number_epochs = 100
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# CONSTRUINDO O AMBIENTE CRIANDO UM OBJETO DE CLASSE DE AMBIENTE
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# CONSTRUINDO O CÉREBRO CRIANDO UM OBJETO DA CLASSE BRAIN
brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)

# CONSTRUINDO O MODELO DQN CRIANDO UM OBJETO DA CLASSE DQN
dqn = dqn.DQN(max_memory = max_memory, discount_factor = 0.9)

# ESCOLHENDO O MODO DE TREINAMENTO
train = True

# TREINE A IA
env.train = train
model = brain.model
early_stoping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0

if (env.train):
    # INICIAR O CICLO DE TODAS AS ÉPOCAS (1 Época = 5 Meses)
    for epoch in range(1, number_epochs):
        # INICIALIZAÇÃO DE VARIÁVEIS DE AMBIENTE E LOOP DE TREINAMENTO
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset(new_month = new_month)  
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        # INICIALIZAÇÃO DO LOOP DE PASSOS DE TEMPO (Passo de tempo = 1 minuto) EM UMA ERA
        while ((not game_over) and (timestep <= 5*30*24*60)):
            # EXECUTE A PRÓXIMA AÇÃO POR SCAN
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                   
            # EXECUTE A SEGUINTE AÇÃO POR INFERÊNCIA
            else: 
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
            
            if (action < direction_boundary):
                direction = -1
            else:
                direction = 1
            energy_ai = abs(action - direction_boundary) * temperature_step
            
            # ATUALIZE O AMBIENTE E CHEGUE AO PRÓXIMO ESTADO
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
            total_reward += reward
            
            # ARMAZENE A NOVA TRANSIÇÃO NA MEMÓRIA
            dqn.remember([current_state, action, reward, next_state], game_over)
            
            # OBTENHA OS DOIS BLOCOS SEPARADOS DE ENTRADAS E OBJETIVOS
            inputs, targets = dqn.get_batch(model, batch_size)
            
            # CALCULE A FUNÇÃO DE PERDA USANDO TODO O BLOCO DE ENTRADA E OS ALVOS
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state
            
        # IMPRIMA OS RESULTADOS DO TREINAMENTO NO FIM DA ÉPOCA
        print("\n")
        print("Epoch: {:03d}/{:03d}.".format(epoch, number_epochs))
        print(" - Total energy spent with AI: {:.0f} J.".format(env.total_energy_ai))
        print(" - Total energy spent with no AI: {:.0f} J.".format(env.total_energy_noai))
        
        # EARLY STOPPING
        if (early_stoping):
            if (total_reward <= best_total_reward):
                patience_count += 1
            elif (total_reward > best_total_reward):
                best_total_reward = total_reward
                patience_count = 0
            if (patience_count > patience):
                print("Early stopping")
                break
        
        # SALVAR O MODELO PARA USO FUTURO
        model.save("model.h5")

