import os
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))

NOMES_ACEITOS = ["dose_radiacao_expandido.csv",
                 "dose_radiacao_expandida.csv",
                 "dose_radiacao_expandido (1).csv"]


def localizar_dataset():
    for nome in NOMES_ACEITOS:
        caminho = os.path.join(PASTA, nome)
        if os.path.exists(caminho):
            return caminho
    raise FileNotFoundError(
        "CSV nao encontrado. Coloque um dos arquivos "
        + " ou ".join(NOMES_ACEITOS) + " na pasta " + PASTA)


ARQUIVO = localizar_dataset()

def ler_csv(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        linhas = [l.strip() for l in f if l.strip()]
    cabecalho = linhas[0].split(",")
    dados = [[float(v) for v in l.split(",")] for l in linhas[1:]]
    matriz = np.array(dados, dtype=float)
    if cabecalho[0].strip() == "":          
        cabecalho = cabecalho[1:]
        matriz = matriz[:, 1:]
    return cabecalho, matriz


def ajustar(X, y, intercepto=True):
    Xp = np.hstack([np.ones((X.shape[0], 1)), X]) if intercepto else X.copy()
    beta = np.linalg.solve(Xp.T @ Xp, Xp.T @ y)
    return beta, Xp


def prever(beta, X, intercepto=True):
    Xp = np.hstack([np.ones((X.shape[0], 1)), X]) if intercepto else X.copy()
    return Xp @ beta


def r2(y, yhat):
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


def r2_ajustado(y, yhat, p):
    n = len(y)
    return 1.0 - (1.0 - r2(y, yhat)) * (n - 1) / (n - p - 1)


def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


def rmse(y, yhat):
    return float(np.sqrt(mse(y, yhat)))


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def resumo_metricas(nome, y, yhat, p):
    print(f"\n  {nome}")
    print(f"    R2          = {r2(y, yhat):.6f}")
    print(f"    R2 ajustado = {r2_ajustado(y, yhat, p):.6f}")
    print(f"    MSE         = {mse(y, yhat):.4f}")
    print(f"    RMSE        = {rmse(y, yhat):.4f}")
    print(f"    MAE         = {mae(y, yhat):.4f}")


def dispersao_ascii(x, y, titulo, rot_x, rot_y, largura=61, altura=17):
    print(f"\n  {titulo}")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    tela = [[" "] * largura for _ in range(altura)]
    if ymin <= 0.0 <= ymax:
        lz = int(round((ymax - 0.0) / (ymax - ymin) * (altura - 1)))
        tela[lz] = ["-"] * largura
    for xi, yi in zip(x, y):
        c = int(round((xi - xmin) / (xmax - xmin) * (largura - 1)))
        l = int(round((ymax - yi) / (ymax - ymin) * (altura - 1)))
        tela[l][c] = "*"
    for i, linha in enumerate(tela):
        if i == 0:
            rot = f"{ymax:9.2f}"
        elif i == altura - 1:
            rot = f"{ymin:9.2f}"
        else:
            rot = " " * 9
        print(f"  {rot} |" + "".join(linha))
    print("  " + " " * 9 + " +" + "-" * largura)
    print("  " + " " * 11 + f"{xmin:<.2f}" +
          " " * max(1, largura - 18) + f"{xmax:.2f}   ({rot_x})")
    print(f"  (eixo vertical: {rot_y})")


def gerar_graficos(y, yhat, e, tempo, pasta):
    try:
        import matplotlib
        matplotlib.use("Agg")          
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [matplotlib nao instalado - apenas os graficos em texto"
              " foram gerados]")
        print("  Para gerar os PNGs: python -m pip install matplotlib")
        return

    arquivos = []


    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(yhat, e, s=18, alpha=0.6, color="#1f77b4",
               edgecolor="black", linewidth=0.3, zorder=3)
    ax.axhline(0.0, color="red", linewidth=1.2, zorder=2)
    ax.set_xlabel("Valores ajustados $\\hat{y}$ (rad)")
    ax.set_ylabel("Residuo $e_i$ (rad)")
    ax.set_title("Residuos x Valores ajustados - modelo completo")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    caminho = os.path.join(pasta, "p2_residuos_x_ajustados.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    arquivos.append(caminho)

  
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(e, bins=20, color="#4c9be8", edgecolor="black")
    ax.set_xlabel("Residuo $e_i$ (rad)")
    ax.set_ylabel("Frequencia")
    ax.set_title("Histograma dos residuos - modelo completo")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    caminho = os.path.join(pasta, "p2_histograma_residuos.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    arquivos.append(caminho)


    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, yhat, s=18, alpha=0.6, color="#1f77b4",
               edgecolor="black", linewidth=0.3, zorder=3)
    lim = [float(min(np.min(y), np.min(yhat))),
           float(max(np.max(y), np.max(yhat)))]
    ax.plot(lim, lim, color="red", linewidth=1.2, zorder=2,
            label="ajuste perfeito (y = $\\hat{y}$)")
    ax.set_xlabel("Dose observada (rad)")
    ax.set_ylabel("Dose ajustada (rad)")
    ax.set_title("Observado x Ajustado - modelo completo")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    caminho = os.path.join(pasta, "p2_observado_x_ajustado.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    arquivos.append(caminho)


    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(tempo, e, s=18, alpha=0.6, color="#1f77b4",
               edgecolor="black", linewidth=0.3, zorder=3)
    ax.axhline(0.0, color="red", linewidth=1.2, zorder=2)
    ax.set_xlabel("Tempo de exposicao (min)")
    ax.set_ylabel("Residuo $e_i$ (rad)")
    ax.set_title("Residuos x Tempo de exposicao - modelo completo")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    caminho = os.path.join(pasta, "p2_residuos_x_tempo.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    arquivos.append(caminho)

    print("\n  Graficos salvos em PNG:")
    for c in arquivos:
        print(f"    - {os.path.basename(c)}")


def main():
    cabecalho, dados = ler_csv(ARQUIVO)
    col = {nome: i for i, nome in enumerate(cabecalho)}

    y = dados[:, col["Dose_de_Radiacao"]]
    corrente = dados[:, col["mAmp"]]
    tempo = dados[:, col["Tempo_de_Exposicao"]]
    n = len(y)

    X = np.column_stack([corrente, tempo])
    nomes = ["mAmp", "Tempo_de_Exposicao"]

    print("=" * 78)
    print("PROJETO 2 - DOSE DE RADIACAO EM CIRCUITOS INTEGRADOS")
    print("=" * 78)
    print(f"\nArquivo    : {os.path.basename(ARQUIVO)}")
    print(f"Observacoes: {n}")
    print("Resposta   : Dose_de_Radiacao (rad)")
    print(f"Regressores: {', '.join(nomes)}")
    print(f"\nEstatisticas descritivas:")
    print(f"  {'variavel':<20}{'media':>12}{'desvio':>12}"
          f"{'minimo':>12}{'maximo':>12}")
    for nome, v in [("Dose_de_Radiacao", y), ("mAmp", corrente),
                    ("Tempo_de_Exposicao", tempo)]:
        print(f"  {nome:<20}{np.mean(v):>12.4f}{np.std(v, ddof=1):>12.4f}"
              f"{np.min(v):>12.4f}{np.max(v):>12.4f}")


    print("\n" + "-" * 78)
    print("(a) MODELO COMPLETO - ajuste por minimos quadrados")
    print("-" * 78)
    beta, _ = ajustar(X, y)
    yhat = prever(beta, X)
    p = X.shape[1]

    print(f"\n  Intercepto (b0)      = {beta[0]:+.6f}")
    print(f"  mAmp (b1)            = {beta[1]:+.6f}")
    print(f"  Tempo_de_Exposicao (b2) = {beta[2]:+.6f}")
    print("\n  Equacao ajustada:")
    print(f"    Dose = {beta[0]:+.4f} {beta[1]:+.4f}*mAmp "
          f"{beta[2]:+.4f}*Tempo")

    print("\n  Interpretacao dos coeficientes (a outra variavel constante):")
    print(f"   - +1 mA de corrente  -> {beta[1]:+.4f} rad na dose esperada")
    print(f"   - +1 minuto de exposicao -> {beta[2]:+.4f} rad na dose esperada")
    print(f"   - Intercepto ({beta[0]:+.4f} rad): dose esperada com corrente 0")
    print("     e tempo 0. Trata-se de uma extrapolacao fora da faixa dos")
    print("     dados (corrente de 10 a 40 mA, tempo de 0,25 a 20 min); o")
    print("     valor negativo indica que a forma aditiva nao descreve bem a")
    print("     regiao proxima da origem.")

   
    print("\n" + "-" * 78)
    print("(b) PREVISAO: corrente = 15 mA, tempo de exposicao = 5 minutos")
    print("-" * 78)
    novo = np.array([[15.0, 5.0]])
    print(f"\n  Dose de radiacao prevista = {prever(beta, novo)[0]:.4f} rad")

    
    print("\n" + "-" * 78)
    print("(c)/(d) R2 E R2 AJUSTADO DO MODELO COMPLETO")
    print("-" * 78)
    print(f"\n  R2          = {r2(y, yhat):.6f}")
    print(f"  R2 ajustado = {r2_ajustado(y, yhat, p):.6f}")
    print(f"""
  Por que usar o R2 ajustado?
  O R2 comum nunca diminui ao se acrescentar um regressor, mesmo que ele nao
  traga informacao: modelos maiores sempre "parecem" melhores. O R2 ajustado
  corrige isso penalizando o numero de parametros (fator (n-1)/(n-p-1)), e so
  cresce quando a variavel nova compensa o grau de liberdade gasto. Por isso
  ele e a estatistica correta para COMPARAR modelos de tamanhos diferentes.
  Com n = {n} e apenas p = {p} regressores a penalizacao e minima
  (R2 = {r2(y, yhat):.4f} contra R2 ajustado = {r2_ajustado(y, yhat, p):.4f});
  a diferenca cresceria muito se o numero de variaveis fosse grande em
  relacao ao numero de observacoes.""")

    
    print("\n" + "-" * 78)
    print("(e) COMPARACAO COM O MODELO ALTERNATIVO (so mAmp)")
    print("-" * 78)
    X_alt = corrente.reshape(-1, 1)
    beta_alt, _ = ajustar(X_alt, y)
    yhat_alt = prever(beta_alt, X_alt)
    print(f"\n  Modelo alternativo: Dose = {beta_alt[0]:+.4f} "
          f"{beta_alt[1]:+.4f}*mAmp")

    resumo_metricas("Modelo completo (mAmp + Tempo)", y, yhat, p)
    resumo_metricas("Modelo alternativo (so mAmp)", y, yhat_alt, 1)

    print(f"""
  Discussao:
  O modelo completo explica {100*r2(y, yhat):.2f}% da variabilidade da dose,
  contra apenas {100*r2(y, yhat_alt):.2f}% do modelo com a corrente sozinha; o
  R2 ajustado ({r2_ajustado(y, yhat, p):.4f} x {r2_ajustado(y, yhat_alt, 1):.4f})
  confirma a vantagem mesmo penalizando o parametro extra, e o RMSE cai de
  {rmse(y, yhat_alt):.2f} rad para {rmse(y, yhat):.2f} rad.
  Nesta amostra, o tempo de exposicao foi a variavel que apresentou maior
  contribuicao para explicar a dose de radiacao: ele varia de 0,25 a 20
  minutos e deixa-lo de fora reduz muito o poder explicativo do modelo.
  O melhor modelo, entre os dois, e o COMPLETO.""")


    print("\n" + "-" * 78)
    print("ANALISE DE RESIDUOS DO MODELO COMPLETO")
    print("-" * 78)
    e = y - yhat
    s = float(np.sqrt(np.sum(e ** 2) / (n - p - 1)))
    print(f"\n  Soma dos residuos     = {np.sum(e):.6e} (deve ser ~0)")
    print(f"  Desvio-padrao         = {np.std(e, ddof=1):.4f}")
    print(f"  Erro padrao da regressao (s) = {s:.4f}")
    print(f"  Residuo minimo/maximo = {np.min(e):.4f} / {np.max(e):.4f}")
    dw = float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))
    print(f"  Durbin-Watson         = {dw:.4f} "
          "(dados experimentais: ver ressalva abaixo)")

    print("\n  Primeiras 20 observacoes (tabela de residuos)")
    print("  " + "-" * 58)
    print(f"  {'i':>4} {'y observado':>14} {'y ajustado':>14} {'residuo e':>14}")
    print("  " + "-" * 58)
    for i in range(min(20, n)):
        print(f"  {i+1:>4} {y[i]:>14.4f} {yhat[i]:>14.4f} {e[i]:>14.4f}")
    print("  " + "-" * 58)
    print(f"  (tabela completa: {n} observacoes)")

    dispersao_ascii(yhat, e, "Residuos x Valores ajustados",
                    "valores ajustados (rad)", "residuo (rad)")
    gerar_graficos(y, yhat, e, tempo, PASTA)
    print(f"""
  Leitura dos graficos:
  - Os residuos apresentam um padrao sistematico (formato curvo / em leque):
    crescem em modulo conforme a dose ajustada aumenta, o que sugere
    variancia nao constante (heterocedasticidade) e um efeito conjunto de
    corrente e tempo nao captado por um modelo puramente aditivo.
  - O valor de Durbin-Watson foi {dw:.4f}. Entretanto, como os dados vem de um
    experimento planejado e a ordem das linhas do arquivo e arbitraria, essa
    metrica deve ser interpretada com cautela.
  - Alternativas que podem ser investigadas para reduzir esse comportamento:
    incluir o termo de interacao (mAmp * Tempo) ou trabalhar com a resposta
    em escala logaritmica.""")

   
    print("\n" + "-" * 78)
    print("(f) MODELO COM INTERCEPTO FORCADO A ZERO")
    print("-" * 78)
    beta0, _ = ajustar(X, y, intercepto=False)
    yhat0 = prever(beta0, X, intercepto=False)
    print(f"\n  mAmp               = {beta0[0]:+.6f}")
    print(f"  Tempo_de_Exposicao = {beta0[1]:+.6f}")

    ss_res0 = float(np.sum((y - yhat0) ** 2))
    r2_0_media = 1.0 - ss_res0 / float(np.sum((y - np.mean(y)) ** 2))
    r2_0_bruto = 1.0 - ss_res0 / float(np.sum(y ** 2))
    print(f"\n  R2 (formula usual, em torno da media) = {r2_0_media:.6f}")
    print(f"  R2 (versao nao centrada, sem media)   = {r2_0_bruto:.6f}")
    print(f"  RMSE                                  = {rmse(y, yhat0):.4f}")
    print(f"  MAE                                   = {mae(y, yhat0):.4f}")
    print(f"\n  Comparacao com o modelo COM intercepto:")
    print(f"    R2   : {r2(y, yhat):.6f} (com)  x  {r2_0_media:.6f} (sem)")
    print(f"    RMSE : {rmse(y, yhat):.4f} (com)  x  {rmse(y, yhat0):.4f} (sem)")

    escolha = "SEM intercepto" if rmse(y, yhat0) <= rmse(y, yhat) else \
              "COM intercepto"
    print(f"""
  Interpretacao pratica:
  Impor intercepto zero significa afirmar que sem corrente ou sem tempo de
  exposicao a dose absorvida e exatamente 0 rad. Diferente do problema do
  arsenio, aqui essa restricao e coerente com o contexto do experimento:
  sem exposicao a raios X nao ha dose absorvida. Ela tambem elimina o
  intercepto negativo estimado no modelo livre ({beta[0]:.2f} rad), que nao
  tem interpretacao pratica.
  Em termos numericos o RMSE passa de {rmse(y, yhat):.2f} para
  {rmse(y, yhat0):.2f} rad. Como o ajuste com intercepto minimiza a soma de
  quadrados sem restricao, o modelo restrito nunca pode ter erro menor - a
  diferenca observada mostra quanto se paga pela restricao.
  Lembrete: o R2 do modelo sem intercepto nao e diretamente comparavel, pois
  a decomposicao da soma de quadrados so vale quando ha intercepto (dai as
  duas versoes apresentadas).
  Escolha: o modelo {escolha} para fins de PREVISAO (menor erro); se o
  objetivo for respeitar a interpretacao teorica do fenomeno, o modelo sem
  intercepto e defensavel, e a perda de precisao e o preco dessa coerencia.""")


    print("\n" + "-" * 78)
    print("(h) ALEM DO R2 - MSE, RMSE E MAE")
    print("-" * 78)
    print(f"\n  {'Modelo':<32}{'MSE':>14}{'RMSE':>12}{'MAE':>12}")
    print("  " + "-" * 70)
    for nome, pred_i in [("Completo (mAmp + Tempo)", yhat),
                         ("Alternativo (so mAmp)", yhat_alt),
                         ("Completo sem intercepto", yhat0)]:
        print(f"  {nome:<32}{mse(y, pred_i):>14.4f}"
              f"{rmse(y, pred_i):>12.4f}{mae(y, pred_i):>12.4f}")

    print(f"""
  Interpretacao:
  - MSE (rad^2) penaliza fortemente os erros grandes; e util para comparar
    modelos, mas sua unidade nao e interpretavel diretamente.
  - RMSE (rad) esta na mesma unidade da resposta: o modelo completo erra em
    media {rmse(y, yhat):.2f} rad, contra {rmse(y, yhat_alt):.2f} rad do
    modelo so com a corrente. Como a dose media observada e
    {np.mean(y):.2f} rad, isso equivale a cerca de
    {100*rmse(y, yhat)/np.mean(y):.1f}% e {100*rmse(y, yhat_alt)/np.mean(y):.1f}%
    da media, respectivamente.
  - MAE e a media dos desvios absolutos ({mae(y, yhat):.2f} rad no modelo
    completo): por nao elevar ao quadrado, e menos sensivel a outliers. O
    RMSE bem maior que o MAE indica que alguns pontos - os de corrente e
    tempo altos - concentram os maiores erros, justamente onde o efeito
    conjunto das duas variaveis nao e captado pelo modelo aditivo.
  Conclusao: em todas as metricas o modelo completo supera o alternativo,
  mas o comportamento dos residuos sugere que um modelo com termo de
  interacao (mAmp * Tempo) poderia representar melhor os dados.""")

    print("\n" + "=" * 78)
    print("FIM - PROJETO 2")
    print("=" * 78)


if __name__ == "__main__":
    main()
