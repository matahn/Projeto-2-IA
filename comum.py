import os
import json
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentes import QLearning, Sarsa


HIPERPARAMETROS = dict(
    alpha=0.3,          # taxa de aprendizado
    gamma=0.99,         # fator de desconto
    epsilon=1.0,        # começa explorando 100%
    epsilon_min=0.01,   # piso de exploração
    epsilon_dec=0.998,  # decaimento multiplicativo por episódio (~0.01 perto do ep. 2300)
    episodes=5000,
)
N_TREINOS = 5                 # execuções de treino independentes (sementes 0..4)
N_TESTES = 100                # episódios de teste com a política gulosa
SEMENTE_TESTE = 10_000        # sementes dos episódios de teste (iguais para todos os agentes)
MAX_PASSOS_TREINO = 200       # limite padrão do Taxi-v4 (TimeLimit)
MAX_PASSOS_TESTE = 1000       # limite generoso no teste para medir quantos passos a tarefa leva
JANELA_MM = 50                # janela da média móvel nos gráficos

AMBIENTES = {
    'deterministico': dict(),
    'chuvoso': dict(is_rainy=True, rainy_probability=0.6),
}
ALGORITMOS = {'Q-learning': QLearning, 'Sarsa': Sarsa}
SLUG = {'Q-learning': 'qlearning', 'Sarsa': 'sarsa'}

DATA_DIR = 'data'
RESULTS_DIR = 'results'


def make_env(ambiente, max_passos=MAX_PASSOS_TREINO):
    return gym.make('Taxi-v4', max_episode_steps=max_passos, **AMBIENTES[ambiente])


# Meta de Reward
def retorno_otimo(ambiente, gamma=0.999, tol=1e-9):
    """Recompensa esperada por episódio da política ótima, calculada com o modelo (env.P).

    Fazemos value iteration para obter a política ótima e depois avaliamos essa política
    sem desconto, a partir da distribuição de estados iniciais. É a "meta" que o agente
    deveria alcançar (média por episódio).
    """
    env = make_env(ambiente).unwrapped
    nS, nA = env.observation_space.n, env.action_space.n
    P = env.P
    V = np.zeros(nS)
    while True:
        Q = np.array([[sum(p * (r + (0 if t else gamma * V[s2])) for p, s2, r, t in P[s][a])
                       for a in range(nA)] for s in range(nS)])
        V_new = Q.max(1)
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    pi = Q.argmax(1)
    # avaliação sem desconto da política ótima
    U = np.zeros(nS)
    for _ in range(100_000):
        U_new = np.array([sum(p * (r + (0 if t else U[s2])) for p, s2, r, t in P[s][pi[s]])
                          for s in range(nS)])
        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U = U_new
    return float(env.initial_state_distrib @ U)


# Treino
def treinar(ambiente, nome_alg, verbose=False):
    """Roda N_TREINOS treinos e devolve (rewards[N, episodes], passos[N, episodes], q_tables[N])."""
    Alg = ALGORITMOS[nome_alg]
    os.makedirs(DATA_DIR, exist_ok=True)
    rewards, passos, qts = [], [], []
    for seed in range(N_TREINOS):
        env = make_env(ambiente)
        agente = Alg(env, seed=seed, **HIPERPARAMETROS)
        arquivo = os.path.join(DATA_DIR, f'q-table-{SLUG[nome_alg]}-{ambiente}-run{seed}.csv')
        agente.train(arquivo, verbose=verbose)
        rewards.append(agente.rewards_per_episode)
        passos.append(agente.actions_per_episode)
        qts.append(agente.q_table.copy())
        env.close()
    return np.array(rewards), np.array(passos), qts


# Teste
def testar(q_table, ambiente, n=N_TESTES):
    """Executa n episódios com a política gulosa. Devolve passos, rewards e sucesso por episódio."""
    env = make_env(ambiente, MAX_PASSOS_TESTE)
    passos, rewards, sucesso = [], [], []
    for k in range(n):
        state, _ = env.reset(seed=SEMENTE_TESTE + k)
        total, t, done, terminated = 0, 0, False, False
        while not done:
            action = int(np.argmax(q_table[state]))
            state, r, terminated, truncated, _ = env.step(action)
            total += r
            t += 1
            done = terminated or truncated
        passos.append(t)
        rewards.append(total)
        sucesso.append(bool(terminated))
    env.close()
    return np.array(passos), np.array(rewards), np.array(sucesso)


# Gráficos
def media_movel(x, janela=JANELA_MM):
    return np.convolve(x, np.ones(janela) / janela, mode='valid')


CORES = {'Q-learning': '#2a6fdb', 'Sarsa': '#e8590c'}
NOME_AMB = {'deterministico': 'determinístico', 'chuvoso': 'chuvoso (is_rainy, p=0.6)'}


def plot_execucoes(treino, metas, ambiente, alg, ax=None, ylim=(-800, 30)):
    """Curva de aprendizado das 5 execuções de um algoritmo + linha da meta."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    r = treino[(ambiente, alg)]['rewards']
    for i, run in enumerate(r):
        ax.plot(np.arange(JANELA_MM, len(run) + 1), media_movel(run), lw=1, alpha=0.8,
                label=f'execução {i + 1}')
    ax.axhline(metas[ambiente], color='k', ls='--', lw=1.5,
               label=f'meta (política ótima) = {metas[ambiente]:.2f}')
    ax.set_ylim(*ylim)
    ax.set_xlabel('Episódio')
    ax.set_ylabel(f'Reward acumulado (média móvel {JANELA_MM})')
    ax.set_title(f'{alg} - ambiente {NOME_AMB[ambiente]}: 5 execuções de treino')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    return ax


def plot_comparacao(treino, metas, ambiente, ax=None, ylim=(-800, 30)):
    """Q-learning vs Sarsa no mesmo gráfico: média das 5 execuções ± desvio padrão."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    for alg in ALGORITMOS:
        mm = np.array([media_movel(run) for run in treino[(ambiente, alg)]['rewards']])
        x = np.arange(JANELA_MM, mm.shape[1] + JANELA_MM)
        m, s = mm.mean(0), mm.std(0)
        ax.plot(x, m, color=CORES[alg], lw=1.8, label=f'{alg} (média de 5 execuções)')
        ax.fill_between(x, m - s, m + s, color=CORES[alg], alpha=0.2)
    ax.axhline(metas[ambiente], color='k', ls='--', lw=1.5,
               label=f'meta (política ótima) = {metas[ambiente]:.2f}')
    ax.set_ylim(*ylim)
    ax.set_xlabel('Episódio')
    ax.set_ylabel(f'Reward acumulado (média móvel {JANELA_MM})')
    ax.set_title(f'Q-learning vs Sarsa - ambiente {NOME_AMB[ambiente]} (faixa = ±1 desvio padrão)')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    return ax


def episodio_convergencia(rewards, meta, tolerancia=1.0):
    """Primeiro episódio em que a média móvel atinge meta - tolerancia (None se nunca)."""
    mm = media_movel(rewards)
    idx = np.where(mm >= meta - tolerancia)[0]
    return int(idx[0] + JANELA_MM) if len(idx) else None




# Persistência
def salvar_resultados(nome, treino, testes):
    """Salva curvas de treino e resultados de teste em results/<nome>.npz.

    treino: {(ambiente, alg): (rewards, passos, q_tables)}
    testes: {(amb_treino, alg, amb_teste): [(passos, rewards, sucesso) por execução]}
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    arrays = {}
    for (amb, alg), (r, p, _) in treino.items():
        arrays[f'treino|{amb}|{alg}|rewards'] = r
        arrays[f'treino|{amb}|{alg}|passos'] = p
    for (amb_tr, alg, amb_te), por_run in testes.items():
        arrays[f'teste|{amb_tr}|{alg}|{amb_te}|passos'] = np.array([x[0] for x in por_run])
        arrays[f'teste|{amb_tr}|{alg}|{amb_te}|rewards'] = np.array([x[1] for x in por_run])
        arrays[f'teste|{amb_tr}|{alg}|{amb_te}|sucesso'] = np.array([x[2] for x in por_run])
    np.savez_compressed(os.path.join(RESULTS_DIR, f'{nome}.npz'), **arrays)


def carregar_resultados(*nomes):
    """Lê e junta os arquivos results/<nome>.npz."""
    treino, testes = {}, {}
    for nome in nomes:
        dados = np.load(os.path.join(RESULTS_DIR, f'{nome}.npz'))
        for k in dados.files:
            partes = k.split('|')
            if partes[0] == 'treino':
                treino.setdefault((partes[1], partes[2]), {})[partes[3]] = dados[k]
            else:
                testes.setdefault((partes[1], partes[2], partes[3]), {})[partes[4]] = dados[k]
    return treino, testes


def carregar_q_tables(ambiente, alg):
    """Carrega as N_TREINOS q-tables salvas em data/ para um ambiente/algoritmo."""
    return [np.loadtxt(os.path.join(DATA_DIR, f'q-table-{SLUG[alg]}-{ambiente}-run{s}.csv'), delimiter=',')
            for s in range(N_TREINOS)]


def figura_com_zoom(funcao_plot, *args, zoom=(-40, 15)):
    """Desenha o gráfico duas vezes: escala completa (esq.) e zoom perto da meta (dir.)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.8))
    funcao_plot(*args, ax=ax1)
    funcao_plot(*args, ax=ax2, ylim=zoom)
    ax2.set_title('zoom perto da meta')
    ax2.get_legend().remove()
    fig.tight_layout()
    return fig


def salvar_figura(fig, nome):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, nome), dpi=120)
    plt.close(fig)


def resumo_teste(d, run=0):
    """Média e desvio padrão de passos/reward e taxa de sucesso de uma execução de teste."""
    p, r, s = d['passos'][run], d['rewards'][run], d['sucesso'][run]
    return dict(passos_media=p.mean(), passos_desvio=p.std(), reward_media=r.mean(),
                reward_desvio=r.std(), sucesso=s.mean())
