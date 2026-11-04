"""
Ambiente determinístico
1) 5 treinos de Q-learning e Sarsa no Taxi-v4 determinístico, registrando o reward por episódio;
2) 100 execuções de teste com a q-table aprendida (média e desvio de timesteps);
3) gráfico único comparando as curvas de aprendizado de Q-learning e Sarsa.
"""
import numpy as np
from comum import (ALGORITMOS, N_TREINOS, treinar, testar, retorno_otimo, salvar_resultados,
                   carregar_resultados, plot_execucoes, plot_comparacao, figura_com_zoom, salvar_figura, resumo_teste, SLUG)

AMBIENTE = 'deterministico'


def main():
    meta = retorno_otimo(AMBIENTE)
    print(f'Meta de reward (política ótima, média por episódio): {meta:.2f}')

    treino, testes = {}, {}
    for alg in ALGORITMOS:
        print(f'Treinando {alg} ({N_TREINOS} execuções)...')
        treino[(AMBIENTE, alg)] = treinar(AMBIENTE, alg)
        testes[(AMBIENTE, alg, AMBIENTE)] = [testar(qt, AMBIENTE) for qt in treino[(AMBIENTE, alg)][2]]
    salvar_resultados('deterministico', treino, testes)

    tr, te = carregar_resultados('deterministico')
    metas = {AMBIENTE: meta}
    for alg in ALGORITMOS:
        salvar_figura(figura_com_zoom(plot_execucoes, tr, metas, AMBIENTE, alg), f'curva-{SLUG[alg]}-{AMBIENTE}.png')
        res = resumo_teste(te[(AMBIENTE, alg, AMBIENTE)])
        print(f'{alg}: timesteps = {res["passos_media"]:.2f} ± {res["passos_desvio"]:.2f} '
              f'| sucesso = {res["sucesso"]:.0%}')
    salvar_figura(figura_com_zoom(plot_comparacao, tr, metas, AMBIENTE), f'comparacao-{AMBIENTE}.png')


if __name__ == '__main__':
    main()
