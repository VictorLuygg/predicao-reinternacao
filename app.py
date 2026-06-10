"""
Triagem de Reinternação Hospitalar — SUS / Alagoas
Aplicação Streamlit para estimativa do risco de reinternação em até 180 dias.
"""

import os

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Configuração ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Triagem de Reinternação — SUS/AL",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Caminhos ───────────────────────────────────────────────────────────────
# Todos os assets (modelo, demo, figuras) ficam dentro de app/
DIR_APP     = os.path.dirname(os.path.abspath(__file__))
DIR_FIGURAS = os.path.join(DIR_APP, "figuras")

# ── Constantes ─────────────────────────────────────────────────────────────
NUMERICAS   = ["IDADE_ANOS", "DIAS_PERM", "N_INTER_PREV"]
CATEGORICAS = ["COMPLEX_DESC", "SEXO_DESC", "OBSTETRICA"]
CAMPOS      = NUMERICAS + CATEGORICAS
ALVO        = "reinternado_180d"
THRESHOLD   = 0.5

COMPLEXIDADES = ["Alta complexidade", "Média complexidade", "Atenção básica", "Não se aplica"]
SEXOS         = ["Masculino", "Feminino"]

NOME_LEGIVEL = {
    "IDADE_ANOS":   "Idade (anos)",
    "DIAS_PERM":    "Dias de permanência",
    "N_INTER_PREV": "Internações prévias",
}


# ── Recursos em cache ──────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    return joblib.load(os.path.join(DIR_APP, "modelo_reinternacao.joblib"))


@st.cache_data
def carregar_demonstracao() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DIR_APP, "demonstracao.csv"))
    df["OBSTETRICA"] = df["OBSTETRICA"].astype(str)
    return df


# ── Utilidades ─────────────────────────────────────────────────────────────
def classificar_risco(prob: float) -> tuple[str, str]:
    if prob < 0.25:
        return "Baixo risco", "#27ae60"
    if prob < 0.40:
        return "Risco moderado", "#f39c12"
    if prob < THRESHOLD:
        return "Risco elevado", "#e67e22"
    return "Alto risco", "#e74c3c"


def gauge_risco(prob: float, label: str, cor: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 44, "color": cor}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar":  {"color": cor, "thickness": 0.28},
            "steps": [
                {"range": [0,  25],  "color": "#d5f5e3"},
                {"range": [25, 40],  "color": "#fef9e7"},
                {"range": [40, 50],  "color": "#fdebd0"},
                {"range": [50, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "#c0392b", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": label, "font": {"size": 18}},
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
    return fig


def importancias_fig(modelo) -> go.Figure:
    nomes_raw = modelo["pre"].get_feature_names_out()
    nomes = [
        NOME_LEGIVEL.get(
            n.replace("num__", "").replace("cat__", ""),
            n.replace("num__", "").replace("cat__", ""),
        )
        for n in nomes_raw
    ]
    imp = pd.Series(modelo["clf"].feature_importances_, index=nomes).sort_values().tail(12)
    fig = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation="h",
        marker_color="#2980b9",
    ))
    fig.update_layout(
        title="Importância das variáveis no modelo",
        xaxis_title="Importância relativa",
        height=380,
        margin=dict(t=40, b=20, l=10, r=20),
    )
    return fig


def histograma_risco(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="risco", nbins=25,
        color_discrete_sequence=["#2980b9"],
        labels={"risco": "Probabilidade estimada de reinternação"},
        title="Distribuição de risco no lote enviado",
    )
    fig.add_vline(
        x=THRESHOLD, line_dash="dash", line_color="red",
        annotation_text="Limiar (50%)", annotation_position="top right",
    )
    fig.update_layout(height=320, margin=dict(t=40, b=20, l=20, r=20))
    return fig


def mostrar_figura(nome: str, legenda: str) -> None:
    caminho = os.path.join(DIR_FIGURAS, nome)
    if os.path.exists(caminho):
        st.image(caminho, caption=legenda, use_container_width=True)


# ── Carregamento dos recursos ──────────────────────────────────────────────
try:
    modelo = carregar_modelo()
    demo   = carregar_demonstracao()
except FileNotFoundError as exc:
    st.error(
        f"**Arquivo não encontrado:** `{exc.filename}`\n\n"
        "Certifique-se de que `modelo_reinternacao.joblib` e `demonstracao.csv` "
        "estão dentro da pasta `app/`."
    )
    st.stop()

# ── Cabeçalho ──────────────────────────────────────────────────────────────
st.title("🏥 Triagem de Reinternação Hospitalar")
st.caption(
    "SUS · Alagoas · Predição de reinternação por qualquer causa em até 180 dias após a alta"
)
st.divider()

aba1, aba2, aba3 = st.tabs(
    ["🔍 Calcular risco", "📂 Processar lote (CSV)", "📊 Sobre o modelo"]
)

# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — Predição individual
# ══════════════════════════════════════════════════════════════════════════════
with aba1:
    col_form, col_res = st.columns([1, 1.6], gap="large")

    with col_form:
        st.subheader("Dados da internação")
        idade        = st.number_input("Idade (anos)", 0, 120, 60)
        dias         = st.number_input("Dias de permanência", 0, 365, 5)
        n_inter      = st.number_input("Internações anteriores no período", 0, 50, 1)
        complexidade = st.selectbox("Complexidade do procedimento", COMPLEXIDADES)
        sexo         = st.radio("Sexo", SEXOS, horizontal=True)
        obstetrica   = st.radio("Internação obstétrica (CID grupo O)", ["Não", "Sim"], horizontal=True)

        st.info(
            "O risco é atualizado automaticamente conforme os campos são preenchidos.",
            icon="ℹ️",
        )

    with col_res:
        st.subheader("Resultado")

        entrada = pd.DataFrame([{
            "IDADE_ANOS":   float(idade),
            "DIAS_PERM":    float(dias),
            "N_INTER_PREV": int(n_inter),
            "COMPLEX_DESC": complexidade,
            "SEXO_DESC":    sexo,
            "OBSTETRICA":   "True" if obstetrica == "Sim" else "False",
        }])

        prob       = float(modelo.predict_proba(entrada[CAMPOS])[:, 1][0])
        label, cor = classificar_risco(prob)

        st.plotly_chart(gauge_risco(prob, label, cor), use_container_width=True)

        m1, m2 = st.columns(2)
        m1.metric("Probabilidade estimada", f"{prob:.1%}")
        m2.metric("Classificação", label)

        st.divider()
        st.plotly_chart(importancias_fig(modelo), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — Processamento em lote
# ══════════════════════════════════════════════════════════════════════════════
with aba2:
    st.subheader("Processar lote de pacientes")

    with st.expander("ℹ️ Formato esperado do arquivo CSV"):
        st.markdown("""
O arquivo deve conter **exatamente estas seis colunas** (nomes sensíveis a maiúsculas/minúsculas):

| Coluna | Tipo | Valores válidos |
|---|---|---|
| `IDADE_ANOS` | Número | 0 – 120 |
| `DIAS_PERM` | Número | ≥ 0 |
| `N_INTER_PREV` | Inteiro | ≥ 0 |
| `COMPLEX_DESC` | Texto | `Alta complexidade` / `Média complexidade` / `Atenção básica` / `Não se aplica` |
| `SEXO_DESC` | Texto | `Masculino` / `Feminino` |
| `OBSTETRICA` | Texto | `True` / `False` |

Colunas adicionais presentes no arquivo são ignoradas. A ordem das colunas não importa.
        """)

        template = pd.DataFrame([
            {
                "IDADE_ANOS": 60, "DIAS_PERM": 5, "N_INTER_PREV": 1,
                "COMPLEX_DESC": "Alta complexidade", "SEXO_DESC": "Masculino",
                "OBSTETRICA": "False",
            },
            {
                "IDADE_ANOS": 35, "DIAS_PERM": 2, "N_INTER_PREV": 0,
                "COMPLEX_DESC": "Atenção básica", "SEXO_DESC": "Feminino",
                "OBSTETRICA": "True",
            },
        ])
        st.download_button(
            "⬇ Baixar modelo de CSV",
            template.to_csv(index=False).encode("utf-8"),
            "modelo_entrada.csv",
            "text/csv",
        )

    arquivo = st.file_uploader("Carregar CSV com os dados das internações", type=["csv"])

    if arquivo:
        lote     = pd.read_csv(arquivo)
        ausentes = [c for c in CAMPOS if c not in lote.columns]

        if ausentes:
            st.error(f"Colunas ausentes no arquivo: `{'`, `'.join(ausentes)}`")
        else:
            lote["OBSTETRICA"]    = lote["OBSTETRICA"].astype(str)
            lote["risco"]         = modelo.predict_proba(lote[CAMPOS])[:, 1]
            lote["previsto"]      = (lote["risco"] >= THRESHOLD).astype(int)
            lote["classificacao"] = lote["risco"].apply(lambda p: classificar_risco(p)[0])
            resultado = lote.sort_values("risco", ascending=False).reset_index(drop=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total de registros", len(resultado))
            m2.metric(
                "Classificados como alto risco (≥ 50%)",
                int((resultado["risco"] >= THRESHOLD).sum()),
            )
            m3.metric("Risco médio estimado no lote", f"{resultado['risco'].mean():.1%}")

            st.plotly_chart(histograma_risco(resultado), use_container_width=True)

            st.dataframe(
                resultado.style
                    .format({"risco": "{:.1%}"})
                    .background_gradient(subset=["risco"], cmap="RdYlGn_r"),
                use_container_width=True,
                height=400,
            )

            st.download_button(
                "⬇ Baixar resultado pontuado",
                resultado.to_csv(index=False).encode("utf-8"),
                "resultado_pontuado.csv",
                "text/csv",
                type="primary",
            )
    else:
        st.info(
            "Carregue um arquivo CSV para iniciar o processamento em lote. "
            "Use o modelo acima para garantir o formato correto.",
            icon="📂",
        )

# ══════════════════════════════════════════════════════════════════════════════
# ABA 3 — Sobre o modelo
# ══════════════════════════════════════════════════════════════════════════════
with aba3:
    st.subheader("Desempenho e interpretação do modelo")
    st.caption(
        "Modelo: Random Forest com balanceamento de classes · "
        "Dados: SIH/SUS — Alagoas (2008–2023) · "
        "Desfecho: reinternação por qualquer causa em até 180 dias após a alta"
    )

    st.markdown("### Desempenho")
    c1, c2 = st.columns(2)
    with c1:
        mostrar_figura("roc.png", "Curva ROC — conjunto de teste")
    with c2:
        mostrar_figura("matriz_confusao.png", "Matriz de confusão")

    st.markdown("### Fatores de risco")
    c3, c4 = st.columns(2)
    with c3:
        mostrar_figura("importancia.png", "Importância das variáveis — Random Forest")
    with c4:
        mostrar_figura("odds_ratio.png", "Odds Ratios — Regressão Logística")

    st.markdown("### Ganho de triagem")
    c5, c6 = st.columns(2)
    with c5:
        mostrar_figura("lift_decil.png", "Taxa de reinternação por faixa de risco (lift)")
    with c6:
        mostrar_figura("shap.png", "SHAP — impacto individual das variáveis por caso")

    st.markdown("### Análise exploratória")
    c7, c8, c9 = st.columns(3)
    with c7:
        mostrar_figura("eda_faixa_etaria.png", "Reinternação por faixa etária")
    with c8:
        mostrar_figura("eda_complexidade.png", "Reinternação por complexidade do procedimento")
    with c9:
        mostrar_figura("eda_historico.png", "Reinternação por número de internações prévias")

    st.divider()
    st.markdown("### Sobre o projeto")
    st.markdown("""
**Pergunta central:** É possível, a partir de dados administrativos disponíveis no momento
da alta hospitalar, identificar pacientes com maior probabilidade de reinternação
por qualquer causa em até 180 dias?

**Variáveis de entrada:** idade, dias de permanência, número de internações prévias,
complexidade do procedimento, sexo e tipo de internação (obstétrica ou não).

**Limitações:** o modelo foi desenvolvido com dados do SIH/SUS de Alagoas e pode não
generalizar para outros estados ou períodos sem recalibração.

Projeto acadêmico — Mestrado · UFAL · 2026
    """)
