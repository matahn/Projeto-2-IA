# Projeto 2: Reinforcement Learning no Taxi-v4 (determinístico e não determinístico)

Agentes **Q-learning** e **Sarsa** treinados no ambiente `Taxi-v4` do Gymnasium, nas versões
determinística e chuvosa (`is_rainy=True, rainy_probability=0.6`).

O relatório completo, está em `relatorio.ipynb`.

# Como executar

pip install -r requirements.txt

python3 exp_deterministico.py   # atividades 1, 2 e 3 (~15 s)
python3 exp_chuvoso.py          # atividades 4 e 5 (~30 s; usa as q-tables do passo anterior)
jupyter notebook relatorio.ipynb


# Estrutura

| Arquivo | Conteúdo |
|---|---|
| `agentes.py` | classes `QLearning` e `Sarsa` (ε-greedy com decaimento, atualização TD) |
| `comum.py` | hiperparâmetros, treino de 5 execuções, teste de 100 episódios, meta de reward por value iteration, gráficos |
| `exp_deterministico.py` | 5 treinos + 100 testes no ambiente determinístico; gráfico comparando os dois algoritmos |
| `exp_chuvoso.py` | q-table determinística testada com chuva; 5 treinos + 100 testes no ambiente chuvoso |
| `relatorio.ipynb` | relatório com gráficos, tabelas, discussões e conclusões |
| `data/` | q-tables aprendidas (`q-table-<algoritmo>-<ambiente>-run<i>.csv`) |
| `results/` | figuras (`.png`) e resultados numéricos (`.npz`) |

# Hiperparâmetros (iguais para os dois algoritmos e os dois ambientes)

| alfa | gama | epslon inicial | epslon mínimo | decaimento de epslon | episódios |
|---|---|---|---|---|---|
| 0,3 | 0,99 | 1,0 | 0,01 | 0,998 por episódio | 5000 |

# Resumo dos resultados (100 episódios de teste, q-table da execução 1)

| Treinado em | Testado em | Q-learning (timesteps) | Sarsa (timesteps) |
|---|---|---|---|
| determinístico | determinístico | 12,95 +- 2,51 | 13,29 +- 2,73 |
| determinístico | chuvoso | 24,76 +- 9,57 | 25,61 +- 10,57 |
| chuvoso | chuvoso | 24,53 +- 8,60 | 26,51 +- 11,06 |

Meta de reward por episódio (política ótima): 7,93 no determinístico e −3,96 no chuvoso.
O Q-learning aprendeu mais rápido e chegou mais perto da meta nos dois ambientes; treinar com chuva
tornou a política robusta (100% de sucesso nas 5 execuções), enquanto as q-tables treinadas sem chuva
às vezes ficam presas contra uma parede quando a chuva tira o táxi da rota.

# Divisão do trabalho

| Membro | Parte | Arquivos |
|---|---|---|
| Vitor Costa | Implementação dos algoritmos e infraestrutura de experimentos | `agentes.py`, `comum.py`, `requirements.txt` |
| Mateus Ahn | Ambiente determinístico (atividades 1, 2 e 3) | `exp_deterministico.py`, `data/*-deterministico-*.csv`, `results/*deterministico*` |
| João Nunes | Ambiente não determinístico (atividades 4 e 5) | `exp_chuvoso.py`, `data/*-chuvoso-*.csv`, `results/*chuvoso*` |
| Carlos Lorenzo | Relatório, discussões e conclusões | `relatorio.ipynb`, `README.md` |
