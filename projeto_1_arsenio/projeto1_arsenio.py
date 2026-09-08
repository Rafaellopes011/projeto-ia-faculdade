import os
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))

NOMES_ACEITOS = ["arsenio_dataset.csv", "arseniodataset.csv"]

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
    return cabecalho, np.array(dados, dtype=float)


def ajustar(X, y, intercepto=True):
    if intercepto:
        Xp = np.hstack([np.ones((X.shape[0], 1)), X])
    else:
        Xp = X.copy()
    XtX = Xp.T @ Xp
    Xty = Xp.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta, Xp


def prever(beta, X, intercepto=True):
    if intercepto:
        Xp = np.hstack([np.ones((X.shape[0], 1)), X])
    else:
        Xp = X.copy()
    return Xp @ beta

def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot

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
    print(f"    MSE         = {mse(y, yhat):.6f}")
    print(f"    RMSE        = {rmse(y, yhat):.6f}")
    print(f"    MAE         = {mae(y, yhat):.6f}")


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
            rot = f"{ymax:9.4f}"
        elif i == altura - 1:
            rot = f"{ymin:9.4f}"
        else:
            rot = " " * 9
        print(f"  {rot} |" + "".join(linha))
    print("  " + " " * 9 + " +" + "-" * largura)
    print("  " + " " * 11 + f"{xmin:<.4f}" +
          " " * max(1, largura - 20) + f"{xmax:.4f}   ({rot_x})")
    print(f"  (eixo vertical: {rot_y})")


def histograma_ascii(v, titulo, n_faixas=9, escala=40):
    print(f"\n  {titulo}")
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if vmax == vmin:
        vmax = vmin + 1.0
    bordas = np.linspace(vmin, vmax, n_faixas + 1)
    for i in range(n_faixas):
        if i == n_faixas - 1:
            cont = int(np.sum((v >= bordas[i]) & (v <= bordas[i + 1])))
        else:
            cont = int(np.sum((v >= bordas[i]) & (v < bordas[i + 1])))
        barra = "#" * int(round(cont / max(1, len(v)) * escala))
        print(f"    [{bordas[i]:8.4f} ; {bordas[i+1]:8.4f})  "
              f"{barra:<{escala}} {cont}")


def gerar_graficos(yhat, e, pasta):
    import matplotlib
    matplotlib.use("Agg")         
    import matplotlib.pyplot as plt

    n = len(e)
    arquivos = []


    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(yhat, e, color="#1f77b4", edgecolor="black", zorder=3)
    ax.axhline(0.0, color="red", linewidth=1.2, zorder=2)
    ax.set_xlabel("Valores ajustados $\\hat{y}$ (ppm)")
    ax.set_ylabel("Residuo $e_i$ (ppm)")
    ax.set_title("Residuos x Valores ajustados - modelo completo")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    caminho = os.path.join(pasta, "p1_residuos_x_ajustados.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    arquivos.append(caminho)


    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(e, bins=9, color="#4c9be8", edgecolor="black")
    ax.set_xlabel("Residuo $e_i$ (ppm)")
    ax.set_ylabel("Frequencia")
    ax.set_title("Histograma dos residuos - modelo completo")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    caminho = os.path.join(pasta, "p1_histograma_residuos.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    arquivos.append(caminho)


    fig, ax = plt.subplots(figsize=(7, 4.5))
    ordem = np.arange(1, n + 1)
    ax.plot(ordem, e, marker="o", linestyle="-", color="#1f77b4", zorder=3)
    ax.axhline(0.0, color="red", linewidth=1.2, zorder=2)
    ax.set_xlabel("Ordem da observacao")
    ax.set_ylabel("Residuo $e_i$ (ppm)")
    ax.set_title("Residuos x Ordem da observacao - modelo completo")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    caminho = os.path.join(pasta, "p1_residuos_x_ordem.png")
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    arquivos.append(caminho)

    print("\n  Graficos salvos em PNG:")
    for c in arquivos:
        print(f"    - {os.path.basename(c)}")


def main():
    cabecalho, dados = ler_csv(ARQUIVO)
    col = {nome: i for i, nome in enumerate(cabecalho)}

    idade = dados[:, col["Idade"]]
    beber = dados[:, col["Uso_Beber"]]
    cozinhar = dados[:, col["Uso_Cozinhar"]]
    agua = dados[:, col["Arsenio_Agua"]]
    y = dados[:, col["Arsenio_Unhas"]]
    n = len(y)

    X_completo = np.column_stack([idade, beber, cozinhar, agua])
    nomes_completo = ["Idade", "Uso_Beber", "Uso_Cozinhar", "Arsenio_Agua"]

    print("=" * 78)
    print("PROJETO 1 - ARSENIO NAS UNHAS DO PE (regressao linear multipla)")
    print("=" * 78)
    print(f"\nArquivo    : {os.path.basename(ARQUIVO)}")
    print(f"Observacoes: {n}")
    print(f"Resposta   : Arsenio_Unhas (ppm)")
    print(f"Regressores: {', '.join(nomes_completo)}")


    print("\n" + "-" * 78)
    print("(a) MODELO COMPLETO - ajuste por minimos quadrados")
    print("-" * 78)
    beta, _ = ajustar(X_completo, y)
    yhat = prever(beta, X_completo)

    print(f"\n  Intercepto (b0) = {beta[0]:+.6f}")
    for nome, b in zip(nomes_completo, beta[1:]):
        print(f"  {nome:<14} = {b:+.6f}")

    print("\n  Equacao ajustada:")
    termos = " ".join(f"{b:+.6f}*{nm}" for nm, b in zip(nomes_completo, beta[1:]))
    print(f"    Arsenio_Unhas = {beta[0]:+.6f} {termos}")

    print("\n  Interpretacao dos coeficientes (mantendo as demais constantes):")
    print(f"   - 1 ano a mais de idade   -> {beta[1]:+.6f} ppm nas unhas")
    print(f"   - +1 categoria de 'beber' -> {beta[2]:+.6f} ppm nas unhas")
    print(f"   - +1 categoria de 'cozinhar' -> {beta[3]:+.6f} ppm nas unhas")
    print(f"   - +1 ppm de arsenio na agua  -> {beta[4]:+.6f} ppm nas unhas")
    print(f"   - Intercepto: valor esperado ({beta[0]:+.6f} ppm) quando todos")
    print("     os regressores valem zero (extrapolacao, sem sentido pratico")
    print("     porque nao existe participante com idade 0 e categorias 0).")


    print("\n" + "-" * 78)
    print("(b) PREVISAO: idade=30, beber=5, cozinhar=5, arsenio na agua=0.135")
    print("-" * 78)
    novo = np.array([[30.0, 5.0, 5.0, 0.135]])
    pred = prever(beta, novo)[0]
    print(f"\n  Arsenio previsto nas unhas = {pred:.6f} ppm")


    print("\n" + "-" * 78)
    print("(d)/(e) R2 E R2 AJUSTADO DO MODELO COMPLETO")
    print("-" * 78)
    p_completo = X_completo.shape[1]
    print(f"\n  R2          = {r2(y, yhat):.6f}")
    print(f"  R2 ajustado = {r2_ajustado(y, yhat, p_completo):.6f}")
    print("""
  Por que usar o R2 ajustado?
  O R2 comum nunca diminui quando um regressor e acrescentado, mesmo que
  esse regressor seja inutil: ele sempre "melhora" com modelos maiores.
  O R2 ajustado penaliza o numero de parametros (n-p-1 no denominador), so
  aumentando quando a variavel nova explica mais do que o custo em graus de
  liberdade. Por isso ele e a estatistica adequada para COMPARAR modelos com
  numeros diferentes de regressores; o R2 comum serve apenas para descrever
  o ajuste de um modelo isolado.
  Observe que o R2 ajustado e menor que o R2 comum (ele nao "e melhor" em
  valor, e mais honesto): a diferenca entre os dois mostra quanto do ajuste
  vem apenas da quantidade de variaveis usadas.""")


    print("\n" + "-" * 78)
    print("(f) COMPARACAO COM O MODELO ALTERNATIVO (so Arsenio_Agua)")
    print("-" * 78)
    X_alt = agua.reshape(-1, 1)
    beta_alt, _ = ajustar(X_alt, y)
    yhat_alt = prever(beta_alt, X_alt)
    print(f"\n  Modelo alternativo: Arsenio_Unhas = {beta_alt[0]:+.6f} "
          f"{beta_alt[1]:+.6f}*Arsenio_Agua")

    resumo_metricas("Modelo completo (4 regressores)", y, yhat, p_completo)
    resumo_metricas("Modelo alternativo (1 regressor)", y, yhat_alt, 1)

    r2c, r2a = r2(y, yhat), r2(y, yhat_alt)
    r2ac = r2_ajustado(y, yhat, p_completo)
    r2aa = r2_ajustado(y, yhat_alt, 1)
    melhor = "COMPLETO" if r2ac > r2aa else "ALTERNATIVO (so Arsenio_Agua)"
    print(f"""
  Discussao:
  O R2 do modelo completo ({r2c:.4f}) e maior que o do alternativo
  ({r2a:.4f}), o que era esperado: acrescentar variaveis nunca reduz o R2.
  A comparacao justa e pelo R2 ajustado: {r2ac:.4f} (completo) contra
  {r2aa:.4f} (alternativo). Pelo criterio do R2 ajustado o melhor modelo e o
  {melhor}. As variaveis idade, uso para beber e uso para cozinhar
  contribuem pouco: nesta amostra, a concentracao de arsenio na agua foi a
  variavel que apresentou maior contribuicao para explicar a concentracao
  encontrada nas unhas.""")


    print("\n" + "-" * 78)
    print("(f-2) ANALISE DE RESIDUOS DO MODELO COMPLETO")
    print("-" * 78)
    e = y - yhat

    print("\n  Tabela de residuos (todas as observacoes)")
    print("  " + "-" * 69)
    print(f"  {'i':>3} {'y observado':>13} {'y ajustado':>13} "
          f"{'residuo e':>13} {'residuo / erro padrao':>23}")
    print("  " + "-" * 69)
    s = float(np.sqrt(np.sum(e ** 2) / (n - p_completo - 1))) 
    for i in range(n):
        print(f"  {i+1:>3} {y[i]:>13.4f} {yhat[i]:>13.4f} "
              f"{e[i]:>13.4f} {e[i]/s:>23.4f}")
    print("  " + "-" * 69)

    print(f"\n  Soma dos residuos      = {np.sum(e):.6e}  (deve ser ~0)")
    print(f"  Media dos residuos     = {np.mean(e):.6e}")
    print(f"  Desvio-padrao amostral = {np.std(e, ddof=1):.6f}")
    print(f"  Erro padrao da regressao (s) = {s:.6f}")
    print(f"  Residuo minimo / maximo = {np.min(e):.4f} / {np.max(e):.4f}")

    z = (e - np.mean(e)) / np.std(e)
    assimetria = float(np.mean(z ** 3))
    curtose = float(np.mean(z ** 4) - 3.0)
    print(f"  Assimetria dos residuos = {assimetria:.4f} (0 = simetrico)")
    print(f"  Curtose (excesso)       = {curtose:.4f} (0 = normal)")

    dw = float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))
    print(f"  Durbin-Watson           = {dw:.4f} "
          "(dados transversais: ver ressalva abaixo)")

    fora = np.where(np.abs(e / s) > 2.0)[0]
    if len(fora):
        print("  Observacoes com |residuo / erro padrao| > 2: "
              + ", ".join(str(i + 1) for i in fora))
    else:
        print("  Nenhuma observacao com |residuo / erro padrao| > 2.")

    dispersao_ascii(yhat, e, "Residuos x Valores ajustados",
                    "valores ajustados", "residuo")
    dispersao_ascii(np.arange(1, n + 1), e, "Residuos x Ordem da observacao",
                    "ordem", "residuo")
    histograma_ascii(e, "Histograma dos residuos")
    gerar_graficos(yhat, e, os.path.dirname(os.path.abspath(__file__)))

    lado = "esquerda" if assimetria < 0 else "direita"
    print(f"""
  Verificacao das suposicoes:
  - Media zero: a soma dos residuos e praticamente nula, como garante o
    metodo dos minimos quadrados com intercepto.
  - Homocedasticidade: no grafico residuos x ajustados a dispersao NAO e
    constante; os pontos com valores ajustados maiores apresentam residuos
    maiores em modulo, indicando variancia crescente (heterocedasticidade).
  - Normalidade: assimetria {assimetria:.2f} e curtose {curtose:.2f} mostram
    residuos assimetricos (cauda a {lado}) e mais pesados que os de uma
    normal; ha observacao com |residuo / erro padrao| acima de 2
    (candidata a outlier / ponto influente).
  - Independencia: o valor de Durbin-Watson foi {dw:.4f}. Entretanto, como os
    dados sao transversais e a ordem dos participantes e arbitraria, essa
    metrica deve ser interpretada com cautela.
  Conclusao: as suposicoes de variancia constante e normalidade sao apenas
  parcialmente atendidas. Uma transformacao da resposta, como o logaritmo de
  Arsenio_Unhas, pode ser investigada para tentar reduzir esse
  comportamento.""")

    print("\n" + "-" * 78)
    print("(g) MODELO COM INTERCEPTO FORCADO A ZERO")
    print("-" * 78)
    beta0, _ = ajustar(X_completo, y, intercepto=False)
    yhat0 = prever(beta0, X_completo, intercepto=False)

    print("\n  Coeficientes (sem intercepto):")
    for nome, b in zip(nomes_completo, beta0):
        print(f"    {nome:<14} = {b:+.6f}")

    ss_res0 = float(np.sum((y - yhat0) ** 2))
    r2_0_media = 1.0 - ss_res0 / float(np.sum((y - np.mean(y)) ** 2))
    r2_0_bruto = 1.0 - ss_res0 / float(np.sum(y ** 2))

    print(f"\n  R2 (formula usual, em torno da media) = {r2_0_media:.6f}")
    print(f"  R2 (versao nao centrada, sem media)   = {r2_0_bruto:.6f}")
    print(f"  RMSE                                  = {rmse(y, yhat0):.6f}")
    print(f"  MAE                                   = {mae(y, yhat0):.6f}")
    print(f"\n  Comparacao com o modelo COM intercepto:")
    print(f"    R2   : {r2(y, yhat):.6f} (com)  x  {r2_0_media:.6f} (sem)")
    print(f"    RMSE : {rmse(y, yhat):.6f} (com)  x  {rmse(y, yhat0):.6f} (sem)")

    escolha = "COM intercepto" if rmse(y, yhat0) > rmse(y, yhat) else \
              "SEM intercepto"
    print(f"""
  Interpretacao pratica:
  Forcar o intercepto a zero significa afirmar que, com idade 0, categorias
  de uso 0 e agua sem arsenio, o arsenio nas unhas seria exatamente 0 ppm.
  Isso e teoricamente insustentavel aqui: idade zero e categoria zero nem
  existem no dominio dos dados (as categorias vao de 1 a 5), e o organismo
  recebe arsenio tambem por outras fontes (alimentos, ar), de modo que a
  resposta nao precisa passar pela origem.
  Alem disso a restricao retira um grau de liberdade ao ajuste: a reta e
  obrigada a passar pela origem e o RMSE aumenta ({rmse(y, yhat):.4f} ->
  {rmse(y, yhat0):.4f}). Atencao: o R2 do modelo sem intercepto nao e
  comparavel diretamente, pois a decomposicao da soma de quadrados so vale
  quando ha intercepto (por isso as duas versoes acima).
  Escolha: o modelo {escolha}, por ter menor erro e por nao impor uma
  restricao sem justificativa fisica.""")

    print("\n" + "-" * 78)
    print("(h) ALEM DO R2 - MSE, RMSE E MAE")
    print("-" * 78)
    print(f"\n  {'Modelo':<34}{'MSE':>12}{'RMSE':>12}{'MAE':>12}")
    print("  " + "-" * 70)
    linhas = [
        ("Completo (4 regressores)", yhat),
        ("Alternativo (so Arsenio_Agua)", yhat_alt),
        ("Completo sem intercepto", yhat0),
    ]
    for nome, pred_i in linhas:
        print(f"  {nome:<34}{mse(y, pred_i):>12.6f}"
              f"{rmse(y, pred_i):>12.6f}{mae(y, pred_i):>12.6f}")

    print(f"""
  Interpretacao:
  - MSE esta em ppm^2 e penaliza fortemente os erros grandes (eleva ao
    quadrado), sendo sensivel a outliers como a observacao de maior
    concentracao.
  - RMSE volta a unidade original (ppm): o modelo completo erra em media
    cerca de {rmse(y, yhat):.4f} ppm, contra {rmse(y, yhat_alt):.4f} ppm do
    modelo so com o arsenio na agua. Como a media observada e
    {np.mean(y):.4f} ppm, o erro tipico e da ordem de
    {100*rmse(y, yhat)/np.mean(y):.1f}% da media.
  - MAE ({mae(y, yhat):.4f} ppm no modelo completo) e a media dos desvios
    absolutos: nao amplifica outliers, por isso e bem menor que o RMSE. A
    distancia entre MAE e RMSE confirma a presenca de poucos erros muito
    grandes, coerente com a analise de residuos.
  Em todas as metricas de erro o modelo completo fica a frente do
  alternativo, mas a vantagem e pequena diante do custo de 3 parametros
  extras - o que reforca a leitura do R2 ajustado.""")

    print("\n" + "=" * 78)
    print("=" * 78)


if __name__ == "__main__":
    main()
