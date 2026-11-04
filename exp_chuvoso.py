"""Experimentos no ambiente chuvoso (is_rainy=True, rainy_probability=0.6).

Executa as q-tables aprendidas no ambiente determinístico 100 vezes no
ambiente chuvoso;
Treina Q-learning e Sarsa diretamente no ambiente chuvoso e testa cada
política em ambos os ambientes.

Requer as q-tables geradas por ``exp_deterministico.py`` em ``data/``.
"""
from comum import (
    ALGORITMOS,
    N_TREINOS,
    carregar_q_tables,
    carregar_resultados,
    figura_com_zoom,
    plot_comparacao,
    plot_execucoes,
    resumo_teste,
    retorno_otimo,
    salvar_figura,
    salvar_resultados,
    SLUG,
    testar,
    treinar,
)


AMBIENTE = 'chuvoso'


def main():
    meta = retorno_otimo(AMBIENTE)
    print(f'Meta de reward no ambiente chuvoso (política ótima): {meta:.2f}')

    treino, testes = {}, {}

    # políticas aprendidas no ambiente determinístico no ambiente chuvoso.
    for alg in ALGORITMOS:
        q_tables = carregar_q_tables('deterministico', alg)
        testes[('deterministico', alg, AMBIENTE)] = [
            testar(q_table, AMBIENTE) for q_table in q_tables
        ]

    # treino no ambiente chuvoso e testes nos dois ambientes.
    for alg in ALGORITMOS:
        print(f'Treinando {alg} no ambiente chuvoso ({N_TREINOS} execuções)...')
        treino[(AMBIENTE, alg)] = treinar(AMBIENTE, alg)
        q_tables = treino[(AMBIENTE, alg)][2]
        testes[(AMBIENTE, alg, AMBIENTE)] = [
            testar(q_table, AMBIENTE) for q_table in q_tables
        ]
        testes[(AMBIENTE, alg, 'deterministico')] = [
            testar(q_table, 'deterministico') for q_table in q_tables
        ]

    salvar_resultados('chuvoso', treino, testes)

    treinamentos, resultados = carregar_resultados('chuvoso')
    metas = {AMBIENTE: meta}
    for alg in ALGORITMOS:
        salvar_figura(
            figura_com_zoom(plot_execucoes, treinamentos, metas, AMBIENTE, alg),
            f'curva-{SLUG[alg]}-{AMBIENTE}.png',
        )
    salvar_figura(
        figura_com_zoom(plot_comparacao, treinamentos, metas, AMBIENTE),
        f'comparacao-{AMBIENTE}.png',
    )

    for (ambiente_treino, alg, ambiente_teste), dados in resultados.items():
        resumo = resumo_teste(dados)
        print(
            f'treino={ambiente_treino:15s} {alg:10s} '
            f'teste={ambiente_teste:15s} '
            f'timesteps = {resumo["passos_media"]:.2f} ± '
            f'{resumo["passos_desvio"]:.2f} | '
            f'sucesso = {resumo["sucesso"]:.0%}'
        )


if __name__ == '__main__':
    main()
