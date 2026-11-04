"""Agentes tabulares Q-learning e Sarsa (baseados no código TaxiDriver da disciplina).

Mudanças em relação à versão original:
- semente aleatória por execução (reprodutibilidade);
- separação entre `terminated` (fim real do episódio) e `truncated` (limite de passos):
  só zeramos o valor do próximo estado quando o episódio realmente terminou;
- decaimento de epsilon com piso em `epsilon_min`;
- o método `train` devolve a q-table e guarda recompensa/passos por episódio.
"""
import numpy as np


class AgenteTabular:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes, seed=None):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha              # taxa de aprendizado
        self.gamma = gamma              # fator de desconto
        self.epsilon = epsilon          # probabilidade de exploração
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.actions_per_episode = []
        self.rewards_per_episode = []

    def select_action(self, state):
        # epsilon-greedy: explora com probabilidade epsilon, senão usa a melhor ação conhecida
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.env.action_space.n))
        return int(np.argmax(self.q_table[state]))

    def _decai_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_dec)

    def train(self, filename=None, verbose=False):
        raise NotImplementedError


class QLearning(AgenteTabular):
    """Off-policy: alvo usa max_a Q(s', a)."""

    def train(self, filename=None, verbose=False):
        for i in range(1, self.episodes + 1):
            state, _ = self.env.reset(seed=None if self.seed is None or i > 1 else self.seed)
            rewards, actions, done = 0, 0, False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_max = 0.0 if terminated else np.max(self.q_table[next_state])
                td_target = reward + self.gamma * next_max
                self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])
                state = next_state
                rewards += reward
                actions += 1
                done = terminated or truncated
            self.actions_per_episode.append(actions)
            self.rewards_per_episode.append(rewards)
            if verbose and i % 500 == 0:
                print(f"Episódio {i} - ações: {actions} - recompensa: {rewards} - eps: {self.epsilon:.3f}")
            self._decai_epsilon()
        if filename is not None:
            np.savetxt(filename, self.q_table, delimiter=',')
        return self.q_table


class Sarsa(AgenteTabular):
    """On-policy: alvo usa Q(s', a') com a' escolhida pela própria política epsilon-greedy."""

    def train(self, filename=None, verbose=False):
        for i in range(1, self.episodes + 1):
            state, _ = self.env.reset(seed=None if self.seed is None or i > 1 else self.seed)
            action = self.select_action(state)  # no Sarsa a primeira ação é escolhida antes do loop
            rewards, actions, done = 0, 0, False
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_action = self.select_action(next_state)
                next_value = 0.0 if terminated else self.q_table[next_state, next_action]
                td_target = reward + self.gamma * next_value
                self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])
                state, action = next_state, next_action
                rewards += reward
                actions += 1
                done = terminated or truncated
            self.actions_per_episode.append(actions)
            self.rewards_per_episode.append(rewards)
            if verbose and i % 500 == 0:
                print(f"Episódio {i} - ações: {actions} - recompensa: {rewards} - eps: {self.epsilon:.3f}")
            self._decai_epsilon()
        if filename is not None:
            np.savetxt(filename, self.q_table, delimiter=',')
        return self.q_table
