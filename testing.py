# Intelig�ncia Artificial aplicada a Neg�cios e Empresas - Caso Pr�tico 2
# Fase de testes


# Importar bibliotecas e outros arquivos python
import os
import numpy as np
import random as rn
from keras.models import load_model
import environment

# Configurar seeds para reprodutibilidade
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# CONFIGURANDO OS PAR�METROS
number_actions = 5
direction_boundary = (number_actions -1)/2
temperature_step = 1.5

# CONSTRUINDO O AMBIENTE CRIANDO UM OBJETO DE CLASSE DE AMBIENTE
env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# CARREGANDO UM MODELO PR�-TREINADO
model = load_model("model.h5")

# ESCOLHENDO O MODO DE TREINAMENTO
train = False

# Executando uma simula��o de um ano no modo de infer�ncia
env.train = train
current_state, _, _ = env.observe()
for timestep in range(0, 12*30*24*60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
            
    if (action < direction_boundary):
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temperature_step
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
    current_state = next_state


            
# IMPRIME OS RESULTADOS DO TREINAMENTO NO FIM DA �POCA
print("\n")
print(" - Total energy spent with AI: {:.0f} J.".format(env.total_energy_ai))
print(" - Total energy spent with no AI: {:.0f} J.".format(env.total_energy_noai))
print("ENERGY SAVED: {:.0f} %.".format(100*(env.total_energy_noai-env.total_energy_ai)/env.total_energy_noai))


