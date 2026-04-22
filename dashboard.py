"""
================================================================
  DASHBOARD — Sistema de Detección de Fraude
  Ingeniería de Software · Ciencia de Datos en Organizaciones
================================================================
Requisitos:
    pip install streamlit plotly pandas pyarrow scikit-learn

Uso:
    streamlit run dashboard.py
================================================================
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── scikit-learn (solo para métricas del modelo) ──
try:
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
        precision_recall_curve, classification_report
    )
    SKLEARN = True
except ImportError:
    SKLEARN = False

# ════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA
# ════════════════════════════════════════════════
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta de colores consistente ──
COLOR_FRAUD    = "#ff6b35"
COLOR_OK       = "#00d4ff"
COLOR_BG       = "#0a0e1a"
COLOR_SURFACE  = "#111827"
COLOR_MUTED    = "#64748b"
PLOTLY_TEMPLATE = "plotly_dark"

# ════════════════════════════════════════════════
#  CSS GLOBAL
# ════════════════════════════════════════════════
st.markdown("""
<style>
  /* fondo general */
  .stApp { background-color: #0a0e1a; }
  section[data-testid="stSidebar"] { background-color: #0f1929; }

  /* KPI cards */
  .kpi-card {
    background: #111827;
    border: 1px solid #1e2d47;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
  }
  .kpi-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
  }
  .kpi-label {
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-top: 4px;
  }
  .kpi-delta {
    font-size: 0.8rem;
    margin-top: 6px;
  }

  /* títulos de sección */
  .section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #00d4ff;
    letter-spacing: .1em;
    text-transform: uppercase;
    border-left: 3px solid #00d4ff;
    padding-left: 10px;
    margin: 28px 0 14px;
  }

  /* quitar padding default de columnas */
  div[data-testid="column"] { padding: 0 6px; }

  /* tablas */
  .dataframe thead th {
    background: #0d1c2e !important;
    color: #00d4ff !important;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  CARGA DE DATOS
# ════════════════════════════════════════════════
# Ruta relativa al script, no al directorio de trabajo.
# Así funciona sin importar desde dónde se ejecute streamlit.
# Streamlit a veces no resuelve __file__ correctamente.
# Usamos una lista de candidatos y tomamos el primero que existe.
_CANDIDATOS = [
    Path(__file__).resolve().parent / "data" / "clean",
    Path(r"C:\Users\delfl\OneDrive\Desktop\Proyecto IS\Integrador_IngSoft\data\clean"),
    Path("data/clean"),
]
CLEAN_DIR = next((p for p in _CANDIDATOS if p.exists()), _CANDIDATOS[0])

@st.cache_data(show_spinner="Cargando transacciones…")
def load_transactions():
    p = CLEAN_DIR / "transactions_clean.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.to_period("M").astype(str)
    return df

@st.cache_data(show_spinner="Cargando tarjetas…")
def load_cards():
    p = CLEAN_DIR / "cards_clean.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(show_spinner="Cargando usuarios…")
def load_users():
    p = CLEAN_DIR / "users_clean.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(show_spinner="Cargando labels…")
def load_labels():
    p = CLEAN_DIR / "fraud_labels_clean.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(show_spinner="Cargando MCC…")
def load_mcc():
    import json
    p = CLEAN_DIR / "mcc_codes_clean.json"
    if not p.exists():
        return pd.DataFrame(columns=["code","description","category_group"])
    return pd.read_json(p)

# ── Cargar todo ──
txn    = load_transactions()
cards  = load_cards()
users  = load_users()
labels = load_labels()
mcc_df = load_mcc()

# ── Merge principal: transactions + labels ──
@st.cache_data(show_spinner="Combinando datos…")
def build_master(_txn, _labels, _cards, _users, _mcc):
    # Solo transactions es obligatorio; labels es opcional
    if _txn is None:
        return None

    df = _txn.copy()

    # Si hay labels las mergeamos, si no is_fraud queda en False
    if _labels is not None:
        df = df.merge(
            _labels.rename(columns={"transaction_id": "id"}),
            on="id", how="left"
        )
    df["is_fraud"] = df.get("is_fraud", False)
    df["is_fraud"] = df["is_fraud"].fillna(False).astype(bool)

    # Agregar MCC description
    if not _mcc.empty and "mcc" in df.columns:
        mcc_map = _mcc.set_index("code")[["description","category_group"]].to_dict("index")
        df["mcc_desc"]  = df["mcc"].astype(str).map(
            lambda x: mcc_map.get(x, {}).get("description", "Desconocido"))
        df["mcc_group"] = df["mcc"].astype(str).map(
            lambda x: mcc_map.get(x, {}).get("category_group", "Otro"))

    # Agregar card_type desde cards
    if _cards is not None and "card_type" in _cards.columns:
        card_map = _cards.set_index("id")["card_type"].to_dict()
        if "card_id" in df.columns:
            df["card_type"] = df["card_id"].map(card_map)

    return df

master = build_master(txn, labels, cards, users, mcc_df)

# ── Datos de demo si no hay archivos ──
if master is None:
    st.warning("⚠️  No se encontraron archivos en `data/clean/`. Mostrando datos de demostración.")
    rng = np.random.default_rng(42)
    n = 50_000
    dates = pd.date_range("2019-01-01", "2023-12-31", periods=n)
    fraud_mask = rng.random(n) < 0.03
    mcc_codes_sample = ["5812","5411","5814","4814","7011","5912","5732",
                        "4829","5311","7995"]
    master = pd.DataFrame({
        "id":        range(n),
        "date":      dates,
        "month":     pd.DatetimeIndex(dates).to_period("M").astype(str),
        "year":      pd.DatetimeIndex(dates).year,
        "amount":    np.where(fraud_mask,
                              rng.uniform(200, 4000, n),
                              rng.uniform(5, 800, n)),
        "is_fraud":  fraud_mask,
        "mcc":       rng.choice(mcc_codes_sample, n),
        "mcc_desc":  rng.choice(["Restaurantes","Supermercados","Fast Food",
                                  "Telecomunicaciones","Hoteles","Farmacias",
                                  "Electrónica","Money Transfer","Tiendas Dept.",
                                  "Casinos"], n),
        "mcc_group": rng.choice(["Retail / Comercio","Hotelería / Entretenimiento",
                                  "Finanzas / Seguros","Servicios Públicos"], n),
        "card_type": rng.choice(["Debit","Credit","Debit (Prepaid)"], n,
                                 p=[0.5, 0.35, 0.15]),
        "use_chip":  rng.choice(["Chip Transaction","Swipe Transaction",
                                  "Online Transaction"], n),
        "merchant_state": rng.choice(["CA","TX","NY","FL","IL","PA","OH"], n),
    })
    if cards is None:
        cards = pd.DataFrame({
            "id": range(500),
            "card_brand": rng.choice(["Visa","Mastercard","Discover"], 500),
            "card_type":  rng.choice(["Debit","Credit","Debit (Prepaid)"], 500),
            "has_chip":   rng.choice([True, False], 500, p=[0.85,0.15]),
            "card_on_dark_web": rng.choice([True,False], 500, p=[0.05,0.95]),
            "credit_limit": rng.uniform(1000,50000,500),
        })
    if users is None:
        users = pd.DataFrame({
            "id": range(200),
            "current_age":   rng.integers(18,80,200),
            "yearly_income": rng.uniform(20000,150000,200),
            "credit_score":  rng.integers(300,850,200),
            "total_debt":    rng.uniform(0,200000,200),
            "gender":        rng.choice(["Male","Female"],200),
        })


# ════════════════════════════════════════════════
#  SIDEBAR — FILTROS
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Filtros")

    # Rango de fechas
    if "date" in master.columns and master["date"].notna().any():
        min_date = master["date"].min().date()
        max_date = master["date"].max().date()
        fecha_rango = st.date_input(
            "Rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if len(fecha_rango) == 2:
            f_ini, f_fin = pd.Timestamp(fecha_rango[0]), pd.Timestamp(fecha_rango[1])
            master = master[master["date"].between(f_ini, f_fin)]
    
    # Tipo de tarjeta
    if "card_type" in master.columns:
        tipos = ["Todos"] + sorted(master["card_type"].dropna().unique().tolist())
        card_sel = st.selectbox("Tipo de tarjeta", tipos)
        if card_sel != "Todos":
            master = master[master["card_type"] == card_sel]

    # Grupo MCC
    if "mcc_group" in master.columns:
        grupos = ["Todos"] + sorted(master["mcc_group"].dropna().unique().tolist())
        mcc_sel = st.selectbox("Categoría MCC", grupos)
        if mcc_sel != "Todos":
            master = master[master["mcc_group"] == mcc_sel]

    # Monto mínimo
    if "amount" in master.columns:
        max_amt = float(master["amount"].quantile(0.99))
        monto_min = st.slider("Monto mínimo ($)", 0.0, max_amt, 0.0, step=10.0)
        master = master[master["amount"] >= monto_min]

    st.markdown("---")
    st.markdown(f"**Transacciones filtradas:** {len(master):,}")
    st.markdown(f"**Fraudes filtrados:** {int(master['is_fraud'].sum()):,}")


# ════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════
st.markdown("""
<div style='margin-bottom:8px'>
  <span style='font-size:11px;letter-spacing:.2em;color:#00d4ff;text-transform:uppercase'>
    // Ingeniería de Software · Ciencia de Datos en Organizaciones
  </span>
</div>
<h1 style='font-size:2rem;font-weight:800;margin:0;line-height:1.1'>
  🚨 Dashboard de <span style='color:#00d4ff'>Detección de Fraude</span>
</h1>
<p style='color:#64748b;font-size:0.85rem;margin-top:6px'>
  Monitoreo en tiempo real · KPIs · Análisis de riesgo · Performance del modelo
</p>
<hr style='border-color:#1e2d47;margin:16px 0'>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  NAVEGACIÓN POR TABS
# ════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 KPIs Generales",
    "🏪 Fraude por Categoría",
    "👤 Perfil de Riesgo",
    "🤖 Performance del Modelo",
])


# ════════════════════════════════════════════════
#  TAB 1 — KPIs GENERALES
# ════════════════════════════════════════════════
with tab1:

    total_txn    = len(master)
    total_fraud  = int(master["is_fraud"].sum())
    fraud_rate   = total_fraud / total_txn * 100 if total_txn else 0
    monto_total  = master["amount"].sum()
    monto_fraude = master.loc[master["is_fraud"], "amount"].sum()
    avg_fraud    = master.loc[master["is_fraud"], "amount"].mean() if total_fraud else 0
    avg_ok       = master.loc[~master["is_fraud"], "amount"].mean() if total_txn - total_fraud else 0

    # ── Fila KPIs ──
    st.markdown('<div class="section-title">KPIs Principales</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    def kpi(col, valor, label, color="#e2e8f0", delta=None):
        delta_html = f'<div class="kpi-delta" style="color:{COLOR_FRAUD}">{delta}</div>' if delta else ""
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:{color}">{valor}</div>
          <div class="kpi-label">{label}</div>
          {delta_html}
        </div>""", unsafe_allow_html=True)

    kpi(k1, f"{total_txn:,}",        "Total transacciones",   COLOR_OK)
    kpi(k2, f"{total_fraud:,}",       "Fraudes detectados",    COLOR_FRAUD)
    kpi(k3, f"{fraud_rate:.2f}%",     "Tasa de fraude",        COLOR_FRAUD,
        "⚠️ Alto" if fraud_rate > 5 else "✓ Normal")
    kpi(k4, f"${monto_fraude:,.0f}",  "Monto en riesgo ($)",   COLOR_FRAUD)
    kpi(k5, f"${avg_fraud:,.0f}",     "Monto medio fraude",    "#f59e0b")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráfico: Evolución temporal ──
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-title">Evolución Temporal</div>', unsafe_allow_html=True)
        if "month" in master.columns:
            ts = (master.groupby(["month","is_fraud"])
                  .agg(count=("id","count"), monto=("amount","sum"))
                  .reset_index())
            ts_fraud = ts[ts["is_fraud"] == True].sort_values("month")
            ts_ok    = ts[ts["is_fraud"] == False].sort_values("month")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=ts_ok["month"], y=ts_ok["count"],
                name="Legítimas", marker_color="#1e3a5f", opacity=0.7
            ), secondary_y=False)
            fig.add_trace(go.Bar(
                x=ts_fraud["month"], y=ts_fraud["count"],
                name="Fraudes", marker_color=COLOR_FRAUD, opacity=0.9
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=ts_fraud["month"], y=ts_fraud["monto"],
                name="Monto fraude ($)", mode="lines+markers",
                line=dict(color="#f59e0b", width=2),
                marker=dict(size=5)
            ), secondary_y=True)
            fig.update_layout(
                template=PLOTLY_TEMPLATE, barmode="stack",
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=0, r=0, t=10, b=0), height=340,
            )
            fig.update_yaxes(title_text="Transacciones", secondary_y=False,
                             gridcolor="#1e2d47")
            fig.update_yaxes(title_text="Monto ($)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">Distribución</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Legítimas", "Fraudulentas"],
            values=[total_txn - total_fraud, total_fraud],
            marker_colors=[COLOR_OK, COLOR_FRAUD],
            hole=0.55,
            textinfo="percent+label",
            textfont_size=12,
        ))
        fig_pie.update_layout(
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            showlegend=False, height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            annotations=[dict(text=f"{fraud_rate:.1f}%<br>fraude",
                              x=0.5, y=0.5, font_size=14,
                              showarrow=False, font_color=COLOR_FRAUD)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Monto legítimo vs fraude
        fig_bar = go.Figure(go.Bar(
            x=["Legítimas","Fraudulentas"],
            y=[monto_total - monto_fraude, monto_fraude],
            marker_color=[COLOR_OK, COLOR_FRAUD],
            text=[f"${(monto_total-monto_fraude):,.0f}",
                  f"${monto_fraude:,.0f}"],
            textposition="auto",
        ))
        fig_bar.update_layout(
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            height=180, margin=dict(l=0,r=0,t=10,b=0),
            showlegend=False,
            yaxis=dict(gridcolor="#1e2d47"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Distribución de montos ──
    st.markdown('<div class="section-title">Distribución de Montos</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=master.loc[~master["is_fraud"], "amount"].clip(upper=2000),
            nbinsx=60, name="Legítimas",
            marker_color=COLOR_OK, opacity=0.7
        ))
        fig_hist.add_trace(go.Histogram(
            x=master.loc[master["is_fraud"], "amount"].clip(upper=2000),
            nbinsx=60, name="Fraudes",
            marker_color=COLOR_FRAUD, opacity=0.8
        ))
        fig_hist.update_layout(
            barmode="overlay", template=PLOTLY_TEMPLATE,
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            height=260, margin=dict(l=0,r=0,t=10,b=0),
            xaxis_title="Monto ($)",
            legend=dict(orientation="h", y=1.1),
            yaxis=dict(gridcolor="#1e2d47"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        if "use_chip" in master.columns:
            chip_fraud = (master.groupby(["use_chip","is_fraud"])
                          .size().reset_index(name="count"))
            fig_chip = px.bar(
                chip_fraud, x="use_chip", y="count",
                color="is_fraud", barmode="group",
                color_discrete_map={True: COLOR_FRAUD, False: COLOR_OK},
                labels={"use_chip":"Tipo uso","count":"Transacciones","is_fraud":"Fraude"},
                template=PLOTLY_TEMPLATE,
            )
            fig_chip.update_layout(
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=260, margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(gridcolor="#1e2d47"),
                legend_title="Fraude",
            )
            st.plotly_chart(fig_chip, use_container_width=True)


# ════════════════════════════════════════════════
#  TAB 2 — FRAUDE POR CATEGORÍA
# ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Fraude por Categoría MCC</div>',
                unsafe_allow_html=True)

    if "mcc_desc" in master.columns:
        mcc_stats = (master.groupby("mcc_desc")
                     .agg(
                         total=("id","count"),
                         fraudes=("is_fraud","sum"),
                         monto_fraude=("amount", lambda x: x[master.loc[x.index,"is_fraud"]].sum()),
                     )
                     .reset_index())
        mcc_stats["tasa_fraude"] = mcc_stats["fraudes"] / mcc_stats["total"] * 100
        mcc_stats = mcc_stats.sort_values("fraudes", ascending=False).head(20)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 20 categorías por cantidad de fraudes**")
            fig_mcc = px.bar(
                mcc_stats.sort_values("fraudes"),
                x="fraudes", y="mcc_desc", orientation="h",
                color="tasa_fraude",
                color_continuous_scale=["#1e3a5f", COLOR_FRAUD],
                labels={"fraudes":"Fraudes","mcc_desc":"Categoría",
                        "tasa_fraude":"Tasa (%)"},
                template=PLOTLY_TEMPLATE,
            )
            fig_mcc.update_layout(
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=480, margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(tickfont=dict(size=10)),
                xaxis=dict(gridcolor="#1e2d47"),
                coloraxis_colorbar=dict(title="Tasa %"),
            )
            st.plotly_chart(fig_mcc, use_container_width=True)

        with c2:
            st.markdown("**Monto en riesgo por categoría ($)**")
            fig_monto = px.bar(
                mcc_stats.sort_values("monto_fraude").tail(15),
                x="monto_fraude", y="mcc_desc", orientation="h",
                color="monto_fraude",
                color_continuous_scale=["#2a1a0e", COLOR_FRAUD],
                labels={"monto_fraude":"Monto ($)","mcc_desc":"Categoría"},
                template=PLOTLY_TEMPLATE,
            )
            fig_monto.update_layout(
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=480, margin=dict(l=0,r=0,t=10,b=0),
                yaxis=dict(tickfont=dict(size=10)),
                xaxis=dict(gridcolor="#1e2d47"),
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_monto, use_container_width=True)

    # ── Treemap por grupo ──
    if "mcc_group" in master.columns:
        st.markdown('<div class="section-title">Mapa de Calor por Grupo de Comercio</div>',
                    unsafe_allow_html=True)
        grp = (master[master["is_fraud"]]
               .groupby(["mcc_group","mcc_desc"] if "mcc_desc" in master.columns else ["mcc_group"])
               .agg(fraudes=("id","count"), monto=("amount","sum"))
               .reset_index())
        if "mcc_desc" in grp.columns:
            fig_tree = px.treemap(
                grp, path=["mcc_group","mcc_desc"],
                values="fraudes", color="monto",
                color_continuous_scale=["#0a0e1a","#ff6b35"],
                labels={"fraudes":"Fraudes","monto":"Monto ($)"},
                template=PLOTLY_TEMPLATE,
            )
        else:
            fig_tree = px.treemap(
                grp, path=["mcc_group"],
                values="fraudes", color="monto",
                color_continuous_scale=["#0a0e1a","#ff6b35"],
                template=PLOTLY_TEMPLATE,
            )
        fig_tree.update_layout(
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            height=380, margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    # ── Mapa por estado ──
    if "merchant_state" in master.columns:
        st.markdown('<div class="section-title">Fraudes por Estado (EE.UU.)</div>',
                    unsafe_allow_html=True)
        state_fraud = (master[master["is_fraud"]]
                       .groupby("merchant_state")
                       .agg(fraudes=("id","count"), monto=("amount","sum"))
                       .reset_index())
        fig_map = px.choropleth(
            state_fraud,
            locations="merchant_state", locationmode="USA-states",
            color="fraudes", scope="usa",
            color_continuous_scale=["#0e2a3f","#ff6b35"],
            labels={"fraudes":"Fraudes","merchant_state":"Estado"},
            template=PLOTLY_TEMPLATE,
            hover_data={"monto": ":,.0f"},
        )
        fig_map.update_layout(
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            geo=dict(bgcolor=COLOR_BG, lakecolor=COLOR_BG,
                     landcolor="#111827", subunitcolor="#1e2d47"),
            height=380, margin=dict(l=0,r=0,t=10,b=0),
            coloraxis_colorbar=dict(title="Fraudes"),
        )
        st.plotly_chart(fig_map, use_container_width=True)


# ════════════════════════════════════════════════
#  TAB 3 — PERFIL DE RIESGO
# ════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Perfil de Tarjetas de Riesgo</div>',
                unsafe_allow_html=True)

    if cards is not None:
        c1, c2, c3 = st.columns(3)

        # Tarjetas en dark web
        dark_web = int(cards["card_on_dark_web"].sum()) if "card_on_dark_web" in cards.columns else 0
        total_cards = len(cards)
        kpi(c1, f"{dark_web:,}", "Tarjetas en dark web", COLOR_FRAUD,
            f"{dark_web/total_cards*100:.1f}% del total" if total_cards else None)

        sin_chip = int((~cards["has_chip"]).sum()) if "has_chip" in cards.columns else 0
        kpi(c2, f"{sin_chip:,}", "Tarjetas sin chip", "#f59e0b")

        if "credit_limit" in cards.columns:
            avg_limit = cards["credit_limit"].mean()
            kpi(c3, f"${avg_limit:,.0f}", "Límite crédito promedio", COLOR_OK)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if "card_brand" in cards.columns:
                brand_counts = cards["card_brand"].value_counts().reset_index()
                brand_counts.columns = ["brand","count"]
                fig_brand = px.pie(
                    brand_counts, names="brand", values="count",
                    color_discrete_sequence=[COLOR_OK,"#7c3aed",COLOR_FRAUD,"#f59e0b"],
                    hole=0.5, template=PLOTLY_TEMPLATE,
                    title="Distribución por marca"
                )
                fig_brand.update_layout(
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    height=300, margin=dict(l=0,r=0,t=40,b=0)
                )
                st.plotly_chart(fig_brand, use_container_width=True)

        with col2:
            if "card_type" in cards.columns and "card_on_dark_web" in cards.columns:
                risk_type = (cards.groupby("card_type")
                             .agg(total=("id","count"),
                                  dark_web=("card_on_dark_web","sum"))
                             .reset_index())
                risk_type["tasa_riesgo"] = risk_type["dark_web"] / risk_type["total"] * 100
                fig_risk = px.bar(
                    risk_type, x="card_type", y="tasa_riesgo",
                    color="tasa_riesgo",
                    color_continuous_scale=["#0e2a3f", COLOR_FRAUD],
                    labels={"card_type":"Tipo","tasa_riesgo":"% en dark web"},
                    template=PLOTLY_TEMPLATE,
                    title="Riesgo por tipo de tarjeta",
                )
                fig_risk.update_layout(
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    height=300, margin=dict(l=0,r=0,t=40,b=0),
                    yaxis=dict(gridcolor="#1e2d47"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown('<div class="section-title">Perfil Demográfico de Usuarios</div>',
                unsafe_allow_html=True)

    if users is not None:
        col1, col2 = st.columns(2)

        with col1:
            if "credit_score" in users.columns:
                fig_score = go.Figure()
                fig_score.add_trace(go.Histogram(
                    x=users["credit_score"], nbinsx=40,
                    marker_color=COLOR_OK, opacity=0.8, name="Credit Score"
                ))
                fig_score.update_layout(
                    title="Distribución Credit Score",
                    template=PLOTLY_TEMPLATE,
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    height=280, margin=dict(l=0,r=0,t=40,b=0),
                    xaxis_title="Score", yaxis_title="Usuarios",
                    yaxis=dict(gridcolor="#1e2d47"),
                )
                st.plotly_chart(fig_score, use_container_width=True)

        with col2:
            if "yearly_income" in users.columns and "total_debt" in users.columns:
                users_plot = users.copy()
                users_plot["ratio_deuda"] = (
                    users_plot["total_debt"] / users_plot["yearly_income"].replace(0, np.nan)
                ).clip(upper=5)
                fig_scatter = px.scatter(
                    users_plot.sample(min(2000, len(users_plot))),
                    x="yearly_income", y="credit_score",
                    color="ratio_deuda",
                    color_continuous_scale=["#10b981","#f59e0b",COLOR_FRAUD],
                    size_max=8, opacity=0.6,
                    labels={"yearly_income":"Ingreso anual ($)",
                            "credit_score":"Credit Score",
                            "ratio_deuda":"Ratio deuda/ingreso"},
                    template=PLOTLY_TEMPLATE,
                    title="Ingreso vs Credit Score (color = ratio deuda)",
                )
                fig_scatter.update_layout(
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    height=280, margin=dict(l=0,r=0,t=40,b=0),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Tabla top usuarios de riesgo
        if "user_id" in master.columns and users is not None:
            st.markdown('<div class="section-title">Top 10 Usuarios con más Fraudes</div>',
                        unsafe_allow_html=True)
            top_users = (master[master["is_fraud"]]
                         .groupby("user_id")
                         .agg(fraudes=("id","count"), monto_total=("amount","sum"))
                         .reset_index()
                         .sort_values("fraudes", ascending=False)
                         .head(10))
            if users is not None and "credit_score" in users.columns:
                top_users = top_users.merge(
                    users[["id","credit_score","yearly_income"]].rename(
                        columns={"id":"user_id"}),
                    on="user_id", how="left"
                )
            st.dataframe(
                top_users.style.background_gradient(
                    subset=["fraudes","monto_total"], cmap="Reds"),
                use_container_width=True, hide_index=True
            )


# ════════════════════════════════════════════════
#  TAB 4 — PERFORMANCE DEL MODELO
# ════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Métricas del Modelo de Detección</div>',
                unsafe_allow_html=True)

    if not SKLEARN:
        st.warning("scikit-learn no instalado. Ejecutá: `pip install scikit-learn`")
    else:
        # Simulamos scores del modelo con una distribución realista
        # En producción esto vendría de PrediccionFraude.parquet
        rng2 = np.random.default_rng(99)
        n = len(master)
        y_true = master["is_fraud"].astype(int).values

        # Score simulado: fraudes tienen scores más altos en promedio
        fraud_scores = np.where(
            y_true == 1,
            rng2.beta(5, 2, n),   # fraudes: distribución sesgada a 1
            rng2.beta(1, 6, n),   # legítimas: distribución sesgada a 0
        )

        threshold = st.slider(
            "Umbral de decisión (fraud score)", 0.0, 1.0, 0.5, 0.01,
            help="Transacciones con score ≥ umbral se clasifican como fraude"
        )
        y_pred = (fraud_scores >= threshold).astype(int)

        # ── Métricas ──
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        prec  = precision_score(y_true, y_pred, zero_division=0)
        rec   = recall_score(y_true, y_pred, zero_division=0)
        f1    = f1_score(y_true, y_pred, zero_division=0)
        acc   = accuracy_score(y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, fraud_scores)
        roc_auc = auc(fpr, tpr)

        prec_curve, rec_curve, _ = precision_recall_curve(y_true, fraud_scores)
        pr_auc = auc(rec_curve, prec_curve)

        m1, m2, m3, m4, m5 = st.columns(5)
        kpi(m1, f"{prec:.3f}",    "Precision",  COLOR_OK)
        kpi(m2, f"{rec:.3f}",     "Recall",     COLOR_OK)
        kpi(m3, f"{f1:.3f}",      "F1-Score",   COLOR_OK)
        kpi(m4, f"{roc_auc:.3f}", "AUC-ROC",    "#7c3aed")
        kpi(m5, f"{pr_auc:.3f}",  "AUC-PR",     "#7c3aed")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        # ── Curva ROC ──
        with col1:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"ROC (AUC={roc_auc:.3f})",
                line=dict(color="#7c3aed", width=2.5)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                name="Random", line=dict(color=COLOR_MUTED, dash="dash", width=1)
            ))
            fig_roc.update_layout(
                title="Curva ROC", template=PLOTLY_TEMPLATE,
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=320, margin=dict(l=0,r=0,t=40,b=0),
                xaxis=dict(title="FPR", gridcolor="#1e2d47"),
                yaxis=dict(title="TPR", gridcolor="#1e2d47"),
                legend=dict(x=0.6, y=0.1),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Curva PR ──
        with col2:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=rec_curve, y=prec_curve, mode="lines",
                name=f"PR (AUC={pr_auc:.3f})",
                line=dict(color=COLOR_FRAUD, width=2.5)
            ))
            fig_pr.update_layout(
                title="Curva Precision-Recall", template=PLOTLY_TEMPLATE,
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=320, margin=dict(l=0,r=0,t=40,b=0),
                xaxis=dict(title="Recall", gridcolor="#1e2d47"),
                yaxis=dict(title="Precision", gridcolor="#1e2d47"),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        # ── Matriz de confusión ──
        with col3:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp],[fn, tp]],
                x=["Pred: Legítima","Pred: Fraude"],
                y=["Real: Legítima","Real: Fraude"],
                colorscale=[[0,"#0e2a3f"],[1,COLOR_FRAUD]],
                text=[[f"TN\n{tn:,}", f"FP\n{fp:,}"],
                      [f"FN\n{fn:,}", f"TP\n{tp:,}"]],
                texttemplate="%{text}",
                textfont=dict(size=14, color="white"),
                showscale=False,
            ))
            fig_cm.update_layout(
                title="Matriz de Confusión", template=PLOTLY_TEMPLATE,
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=320, margin=dict(l=0,r=0,t=40,b=0),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── Distribución del score ──
        st.markdown('<div class="section-title">Distribución del Fraud Score</div>',
                    unsafe_allow_html=True)
        fig_score_dist = go.Figure()
        fig_score_dist.add_trace(go.Histogram(
            x=fraud_scores[y_true == 0],
            nbinsx=60, name="Legítimas",
            marker_color=COLOR_OK, opacity=0.7
        ))
        fig_score_dist.add_trace(go.Histogram(
            x=fraud_scores[y_true == 1],
            nbinsx=60, name="Fraudes",
            marker_color=COLOR_FRAUD, opacity=0.8
        ))
        fig_score_dist.add_vline(
            x=threshold, line_dash="dash",
            line_color="#f59e0b", line_width=2,
            annotation_text=f"Umbral={threshold:.2f}",
            annotation_font_color="#f59e0b",
        )
        fig_score_dist.update_layout(
            barmode="overlay", template=PLOTLY_TEMPLATE,
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            xaxis_title="Fraud Score", yaxis_title="Cantidad",
            legend=dict(orientation="h", y=1.1),
            yaxis=dict(gridcolor="#1e2d47"),
        )
        st.plotly_chart(fig_score_dist, use_container_width=True)

        # Nota sobre scores reales
        st.info(
            "💡 **Nota:** los fraud scores mostrados son simulados para visualización. "
            "Cuando entrenes tu modelo, reemplazá `fraud_scores` con los valores reales "
            "de la tabla `PrediccionFraude` (columna `fraud_score`)."
        )

# ── Footer ──
st.markdown("""
<hr style='border-color:#1e2d47;margin-top:40px'>
<p style='text-align:center;color:#334155;font-size:11px'>
  Sistema de Detección de Fraude · Dashboard v1.0 · 
  Ingeniería de Software · Ciencia de Datos en Organizaciones · 2026
</p>
""", unsafe_allow_html=True)
