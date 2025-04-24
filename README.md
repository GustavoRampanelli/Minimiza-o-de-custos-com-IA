# 🧠 Inteligência Artificial Aplicada à Gestão Energética

Este projeto demonstra o uso de redes neurais com *Deep Q-Learning* para otimizar o consumo energético de um sistema de climatização. A IA aprende a controlar a temperatura de forma eficiente, economizando energia em relação a um sistema tradicional.

---

## 📁 Estrutura do Projeto
 ├── brain.py # Modelo neural e compilação 
 ├── dqn.py # Implementação da Deep Q-Network 
 ├── environment.py # Ambiente de simulação 
 ├── training.py # Fase de treinamento do agente 
 ├── testing.py # Fase de teste (inferência) 
 ├── model.h5 # Modelo treinado (gerado após treino) 
 └── README.md # Documentação do projeto


---

## ⚙️ Requisitos

- Python 3.7+
- Keras
- TensorFlow
- NumPy

### Instalação rápida com `pip`

bash
pip install numpy tensorflow keras

## 🚀 Como Executar
1. Treinamento
Executa o treinamento da IA em ciclos de aprendizado com base em recompensas energéticas.
python training.py
O modelo será salvo como model.h5 após cada época (ou apenas o melhor, se early stopping estiver ativado).

2. Teste / Simulação
Após o treinamento, execute o modelo em modo inferência para simular o desempenho:
python testing.py
## 🔍 Parâmetros Importantes

Parâmetro        Valor Padrão	        Descrição
number_actions	     5	          Ações disponíveis para o agente
epsilon            	0.3	          Taxa de exploração (ε-greedy)
temperature_step	  1.5	          Intervalo de mudança de temperatura
number_epochs	      100	          Total de épocas para treinamento
max_memory	        3000	        Tamanho da memória de replay
batch_size	        512	          Tamanho dos lotes de treino
early_stopping	    True	        Interrompe se não houver melhora

## 📊 Métricas de Avaliação
Gasto total com IA vs sem IA (em Joules)

% de economia de energia

Recompensa total por época

Perda (loss) média por época

📌 Objetivo
Demonstrar a aplicação de IA para eficiência energética, simulação de cenários de consumo, e uso de aprendizado por reforço em ambientes reais.

📖 Créditos
Projeto desenvolvido como parte da disciplina de Inteligência Artificial Aplicada a Negócios e Empresas.
