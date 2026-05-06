"""
================================================================
  DASHBOARD v2 — Sistema de Detección de Fraude + Análisis por Usuario
  Ingeniería de Software · Ciencia de Datos en Organizaciones
================================================================

Diferencias con el dashboard.py original:
  1. Lee de SQLite (fintrack.db) en vez de archivos parquet.
  2. Agrega el tab "👤 Análisis por Usuario" con mapa, métricas y patrones.
  3. Agrega el tab "➕ Cargar Transacción" para alta en tiempo real.
  4. Las métricas se actualizan al instante cuando se agrega una
     transacción nueva, sin necesidad de reprocesar el CSV.

Requisitos:
    pip install streamlit plotly pandas scikit-learn

Antes de correrlo:
    1. python carga_inicial.py    # genera fintrack.db (una sola vez)

Para correrlo en paralelo con el dashboard viejo, usar puerto distinto:
    streamlit run dashboard2.py --server.port 8502

(El dashboard original queda en el puerto 8501 por default)
================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import sqlite3
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# scikit-learn (solo para métricas del modelo)
try:
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
        precision_recall_curve,
        precision_score, recall_score, f1_score, accuracy_score
    )
    SKLEARN = True
except ImportError:
    SKLEARN = False


# ════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA
# ════════════════════════════════════════════════
st.set_page_config(
    page_title="FinTrack v2 — Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de colores (idéntica al dashboard original)
COLOR_FRAUD    = "#ff6b35"
COLOR_OK       = "#00d4ff"
COLOR_BG       = "#0a0e1a"
COLOR_SURFACE  = "#111827"
COLOR_MUTED    = "#64748b"
COLOR_WARN     = "#f59e0b"
COLOR_USER     = "#7c3aed"
PLOTLY_TEMPLATE = "plotly_dark"


# ════════════════════════════════════════════════
#  CSS GLOBAL (mismo que el original)
# ════════════════════════════════════════════════
st.markdown("""
<style>
  .stApp { background-color: #0a0e1a; }
  section[data-testid="stSidebar"] { background-color: #0f1929; }
  .kpi-card {
    background: #111827;
    border: 1px solid #1e2d47;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
  }
  .kpi-value { font-size: 1.8rem; font-weight: 800; line-height: 1.1; }
  .kpi-label {
    font-size: 0.75rem; color: #64748b;
    letter-spacing: .08em; text-transform: uppercase; margin-top: 4px;
  }
  .kpi-delta { font-size: 0.8rem; margin-top: 6px; }
  .section-title {
    font-size: 1rem; font-weight: 700; color: #00d4ff;
    letter-spacing: .1em; text-transform: uppercase;
    border-left: 3px solid #00d4ff; padding-left: 10px;
    margin: 28px 0 14px;
  }
  div[data-testid="column"] { padding: 0 6px; }
  .dataframe thead th {
    background: #0d1c2e !important;
    color: #00d4ff !important;
  }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  CAPA DE DATOS — TODA LECTURA DE SQLITE PASA POR ACÁ
# ════════════════════════════════════════════════
DB_PATH = Path(__file__).resolve().parent / "fintrack.db"
if not DB_PATH.exists():
    DB_PATH = Path("./fintrack.db")


def _conn():
    """Abre una conexión a la base. Cada función la cierra al terminar."""
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner="Cargando transacciones desde SQLite…")
def load_transactions(_cache_key=0):
    """
    Carga las transacciones, hace el merge con labels de fraude y MCC,
    y devuelve un DataFrame listo para usar.

    El parámetro _cache_key sirve para invalidar el cache cuando se
    agregan transacciones nuevas (RF-02). Cada vez que cambia, se recarga.
    """
    conn = _conn()
    df = pd.read_sql_query("""
        SELECT
            t.transaction_id AS id,
            t.date,
            t.client_id,
            t.card_id,
            t.amount,
            t.use_chip,
            t.merchant_id,
            t.merchant_city,
            t.merchant_state,
            t.merchant_lat,
            t.merchant_lon,
            t.zip,
            t.mcc,
            t.year,
            t.month,
            COALESCE(f.is_fraud, 0) AS is_fraud,
            m.descripcion AS mcc_desc,
            m.grupo AS mcc_group,
            c.card_type
        FROM transactions t
        LEFT JOIN fraud_labels f ON f.transaction_id = t.transaction_id
        LEFT JOIN mcc_codes m    ON m.mcc = t.mcc
        LEFT JOIN cards c        ON c.card_id = t.card_id
    """, conn, parse_dates=["date"])
    conn.close()

    df["is_fraud"] = df["is_fraud"].astype(bool)
    df["mcc_desc"]  = df["mcc_desc"].fillna("Desconocido")
    df["mcc_group"] = df["mcc_group"].fillna("Otro")
    return df


@st.cache_data(show_spinner="Cargando usuarios…")
def load_users():
    conn = _conn()
    df = pd.read_sql_query(
        "SELECT user_id AS id, * FROM users", conn
    )
    df = df.loc[:, ~df.columns.duplicated()]
    conn.close()
    return df


@st.cache_data(show_spinner="Cargando tarjetas…")
def load_cards():
    conn = _conn()
    df = pd.read_sql_query(
        "SELECT card_id AS id, * FROM cards", conn
    )
    df = df.loc[:, ~df.columns.duplicated()]
    # Convertir a booleanos para que los filtros funcionen igual que antes
    if "has_chip" in df.columns:
        df["has_chip"] = df["has_chip"].astype(bool)
    if "card_on_dark_web" in df.columns:
        df["card_on_dark_web"] = df["card_on_dark_web"].astype(bool)
    conn.close()
    return df


# ════════════════════════════════════════════════
#  FUNCIÓN DE INSERCIÓN INCREMENTAL (RF-02)
#  Esta es la magia: agrega una transacción y actualiza
#  los agregados sin releer todo el CSV.
# ════════════════════════════════════════════════

def agregar_transaccion_db(tx_dict):
    """
    Inserta una transacción nueva en la base y actualiza los agregados
    afectados (usuario, comercio, global) de forma incremental.
    Retorna {ok, mensaje}.
    """
    conn = _conn()
    try:
        cur = conn.cursor()

        # Insertar la transacción
        cur.execute("""
            INSERT INTO transactions
                (transaction_id, date, client_id, card_id, amount, use_chip,
                 merchant_id, merchant_city, merchant_state,
                 merchant_lat, merchant_lon, zip, mcc, year, month)
            VALUES (:transaction_id, :date, :client_id, :card_id, :amount, :use_chip,
                    :merchant_id, :merchant_city, :merchant_state,
                    :merchant_lat, :merchant_lon, :zip, :mcc, :year, :month)
        """, tx_dict)

        # Actualizar agregado del usuario
        cur.execute("""
            UPDATE agg_por_usuario
            SET cant_transacciones  = cant_transacciones + 1,
                gasto_total         = gasto_total + :amount,
                gasto_promedio      = (gasto_promedio * cant_transacciones + :amount)
                                       / (cant_transacciones + 1),
                gasto_minimo        = MIN(COALESCE(gasto_minimo, :amount), :amount),
                gasto_maximo        = MAX(COALESCE(gasto_maximo, :amount), :amount),
                ultima_transaccion  = MAX(COALESCE(ultima_transaccion, :date), :date),
                primera_transaccion = MIN(COALESCE(primera_transaccion, :date), :date)
            WHERE user_id = :client_id
        """, tx_dict)

        # Actualizar agregado del comercio (UPSERT)
        cur.execute("""
            INSERT INTO agg_por_comercio
                (merchant_id, merchant_city, merchant_state,
                 cant_transacciones, monto_total, monto_promedio)
            VALUES (:merchant_id, :merchant_city, :merchant_state, 1, :amount, :amount)
            ON CONFLICT(merchant_id) DO UPDATE SET
                cant_transacciones = cant_transacciones + 1,
                monto_total        = monto_total + excluded.monto_total,
                monto_promedio     = (monto_promedio * cant_transacciones + excluded.monto_total)
                                      / (cant_transacciones + 1)
        """, tx_dict)

        # Actualizar agregado global
        cur.execute("""
            UPDATE agg_global
            SET cant_transacciones = cant_transacciones + 1,
                monto_total        = monto_total + :amount,
                monto_promedio     = (monto_promedio * cant_transacciones + :amount)
                                      / (cant_transacciones + 1)
        """, tx_dict)

        conn.commit()
        return {"ok": True, "mensaje": f"✅ Transacción {tx_dict['transaction_id']} agregada."}
    except sqlite3.IntegrityError as e:
        conn.rollback()
        return {"ok": False, "mensaje": f"❌ Error: ya existe una transacción con ese ID. Detalle: {e}"}
    except Exception as e:
        conn.rollback()
        return {"ok": False, "mensaje": f"❌ Error inesperado: {e}"}
    finally:
        conn.close()


# ════════════════════════════════════════════════
#  ESTADO DE SESIÓN — para invalidar cache al insertar
# ════════════════════════════════════════════════
if "cache_key" not in st.session_state:
    st.session_state.cache_key = 0


# ════════════════════════════════════════════════
#  VERIFICACIÓN: ¿existe la base?
# ════════════════════════════════════════════════
if not DB_PATH.exists():
    st.error(f"""
    ❌ **No se encontró la base de datos** `{DB_PATH}`.

    Antes de correr este dashboard, generá la base ejecutando:
    ```
    python carga_inicial.py
    ```
    Eso lee los CSV de `data/`, limpia los datos y crea `fintrack.db`.
    """)
    st.stop()


# ════════════════════════════════════════════════
#  CARGA DE DATOS
# ════════════════════════════════════════════════
master_full = load_transactions(_cache_key=st.session_state.cache_key)
users  = load_users()
cards  = load_cards()


# ════════════════════════════════════════════════
#  SIDEBAR — FILTROS
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Filtros")

    master = master_full.copy()

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
        if isinstance(fecha_rango, tuple) and len(fecha_rango) == 2:
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
    if "amount" in master.columns and len(master) > 0:
        max_amt = float(master["amount"].quantile(0.99))
        monto_min = st.slider("Monto mínimo ($)", 0.0, max_amt, 0.0, step=10.0)
        master = master[master["amount"] >= monto_min]

    st.markdown("---")
    st.markdown(f"**Transacciones filtradas:** {len(master):,}")
    st.markdown(f"**Fraudes filtrados:** {int(master['is_fraud'].sum()):,}")

    st.markdown("---")
    st.caption("💾 Datos: SQLite (fintrack.db)")
    st.caption(f"🔄 Cache key: {st.session_state.cache_key}")


# ════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════
st.markdown("""
<div style='margin-bottom:8px'>
  <span style='font-size:11px;letter-spacing:.2em;color:#00d4ff;text-transform:uppercase'>
    // FinTrack v2 · Procesamiento incremental con SQLite
  </span>
</div>
<h1 style='font-size:2rem;font-weight:800;margin:0;line-height:1.1'>
  🚨 Dashboard de <span style='color:#00d4ff'>Detección de Fraude</span>
  <span style='color:#7c3aed'>+ Análisis por Usuario</span>
</h1>
<p style='color:#64748b;font-size:0.85rem;margin-top:6px'>
  KPIs · Análisis de riesgo · Performance del modelo · Comportamiento individual · Tiempo real
</p>
<hr style='border-color:#1e2d47;margin:16px 0'>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  HELPER: KPI cards
# ════════════════════════════════════════════════
def kpi(col, valor, label, color="#e2e8f0", delta=None):
    delta_html = (
        f'<div class="kpi-delta" style="color:{COLOR_FRAUD}">{delta}</div>'
        if delta else ""
    )
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-value" style="color:{color}">{valor}</div>
      <div class="kpi-label">{label}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  TABS — 6 en total
# ════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 KPIs Generales",
    "🏪 Fraude por Categoría",
    "💳 Perfil de Riesgo",
    "🤖 Performance del Modelo",
    "👤 Análisis por Usuario",
    "➕ Cargar Transacción",
])


# ════════════════════════════════════════════════
#  TAB 1 — KPIs GENERALES (idéntico al original)
# ════════════════════════════════════════════════
with tab1:
    total_txn    = len(master)
    total_fraud  = int(master["is_fraud"].sum())
    fraud_rate   = total_fraud / total_txn * 100 if total_txn else 0
    monto_total  = master["amount"].sum()
    monto_fraude = master.loc[master["is_fraud"], "amount"].sum()
    avg_fraud    = master.loc[master["is_fraud"], "amount"].mean() if total_fraud else 0
    avg_ok       = master.loc[~master["is_fraud"], "amount"].mean() if (total_txn - total_fraud) else 0

    st.markdown('<div class="section-title">KPIs Principales</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)

    kpi(k1, f"{total_txn:,}",        "Total transacciones",   COLOR_OK)
    kpi(k2, f"{total_fraud:,}",       "Fraudes detectados",    COLOR_FRAUD)
    kpi(k3, f"{fraud_rate:.2f}%",     "Tasa de fraude",        COLOR_FRAUD,
        "⚠️ Alto" if fraud_rate > 5 else "✓ Normal")
    kpi(k4, f"${monto_fraude:,.0f}",  "Monto en riesgo ($)",   COLOR_FRAUD)
    kpi(k5, f"${avg_fraud:,.0f}",     "Monto medio fraude",    COLOR_WARN)

    st.markdown("<br>", unsafe_allow_html=True)

    # Evolución temporal
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown('<div class="section-title">Evolución Temporal</div>', unsafe_allow_html=True)
        if "month" in master.columns and len(master) > 0:
            ts = (master.groupby(["month", "is_fraud"])
                  .agg(count=("id", "count"), monto=("amount", "sum"))
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
                line=dict(color=COLOR_WARN, width=2),
                marker=dict(size=5)
            ), secondary_y=True)
            fig.update_layout(
                template=PLOTLY_TEMPLATE, barmode="stack",
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=0, r=0, t=10, b=0), height=340,
            )
            fig.update_yaxes(title_text="Transacciones", secondary_y=False, gridcolor="#1e2d47")
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

        fig_bar = go.Figure(go.Bar(
            x=["Legítimas", "Fraudulentas"],
            y=[monto_total - monto_fraude, monto_fraude],
            marker_color=[COLOR_OK, COLOR_FRAUD],
            text=[f"${(monto_total-monto_fraude):,.0f}", f"${monto_fraude:,.0f}"],
            textposition="auto",
        ))
        fig_bar.update_layout(
            template=PLOTLY_TEMPLATE,
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            height=180, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False, yaxis=dict(gridcolor="#1e2d47"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Distribución de montos
    st.markdown('<div class="section-title">Distribución de Montos</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if len(master) > 0:
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
                height=260, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Monto ($)",
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(gridcolor="#1e2d47"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        if "use_chip" in master.columns and len(master) > 0:
            chip_fraud = (master.groupby(["use_chip", "is_fraud"])
                          .size().reset_index(name="count"))
            fig_chip = px.bar(
                chip_fraud, x="use_chip", y="count",
                color="is_fraud", barmode="group",
                color_discrete_map={True: COLOR_FRAUD, False: COLOR_OK},
                labels={"use_chip": "Tipo uso", "count": "Transacciones", "is_fraud": "Fraude"},
                template=PLOTLY_TEMPLATE,
            )
            fig_chip.update_layout(
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=260, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(gridcolor="#1e2d47"),
                legend_title="Fraude",
            )
            st.plotly_chart(fig_chip, use_container_width=True)


# ════════════════════════════════════════════════
#  TAB 2 — FRAUDE POR CATEGORÍA
# ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Fraude por Categoría MCC</div>', unsafe_allow_html=True)

    if "mcc_desc" in master.columns and len(master) > 0:
        mcc_stats = (master.groupby("mcc_desc")
                     .agg(
                         total=("id", "count"),
                         fraudes=("is_fraud", "sum"),
                         monto_fraude=("amount",
                                       lambda x: x[master.loc[x.index, "is_fraud"]].sum()),
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
                labels={"fraudes": "Fraudes", "mcc_desc": "Categoría", "tasa_fraude": "Tasa (%)"},
                template=PLOTLY_TEMPLATE,
            )
            fig_mcc.update_layout(
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=480, margin=dict(l=0, r=0, t=10, b=0),
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
                labels={"monto_fraude": "Monto ($)", "mcc_desc": "Categoría"},
                template=PLOTLY_TEMPLATE,
            )
            fig_monto.update_layout(
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=480, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(tickfont=dict(size=10)),
                xaxis=dict(gridcolor="#1e2d47"),
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig_monto, use_container_width=True)

    # Treemap por grupo
    if "mcc_group" in master.columns and master["is_fraud"].any():
        st.markdown('<div class="section-title">Mapa de Calor por Grupo de Comercio</div>',
                    unsafe_allow_html=True)
        grp = (master[master["is_fraud"]]
               .groupby(["mcc_group", "mcc_desc"])
               .agg(fraudes=("id", "count"), monto=("amount", "sum"))
               .reset_index())
        fig_tree = px.treemap(
            grp, path=["mcc_group", "mcc_desc"],
            values="fraudes", color="monto",
            color_continuous_scale=["#0a0e1a", "#ff6b35"],
            labels={"fraudes": "Fraudes", "monto": "Monto ($)"},
            template=PLOTLY_TEMPLATE,
        )
        fig_tree.update_layout(
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            height=380, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    # Mapa por estado
    if "merchant_state" in master.columns and master["is_fraud"].any():
        st.markdown('<div class="section-title">Fraudes por Estado (EE.UU.)</div>',
                    unsafe_allow_html=True)
        state_fraud = (master[master["is_fraud"]]
                       .groupby("merchant_state")
                       .agg(fraudes=("id", "count"), monto=("amount", "sum"))
                       .reset_index())
        fig_map = px.choropleth(
            state_fraud,
            locations="merchant_state", locationmode="USA-states",
            color="fraudes", scope="usa",
            color_continuous_scale=["#0e2a3f", "#ff6b35"],
            labels={"fraudes": "Fraudes", "merchant_state": "Estado"},
            template=PLOTLY_TEMPLATE,
            hover_data={"monto": ":,.0f"},
        )
        fig_map.update_layout(
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            geo=dict(bgcolor=COLOR_BG, lakecolor=COLOR_BG,
                     landcolor="#111827", subunitcolor="#1e2d47"),
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title="Fraudes"),
        )
        st.plotly_chart(fig_map, use_container_width=True)


# ════════════════════════════════════════════════
#  TAB 3 — PERFIL DE RIESGO
# ════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Perfil de Tarjetas de Riesgo</div>', unsafe_allow_html=True)

    if cards is not None and len(cards) > 0:
        c1, c2, c3 = st.columns(3)

        dark_web = int(cards["card_on_dark_web"].sum()) if "card_on_dark_web" in cards.columns else 0
        total_cards = len(cards)
        kpi(c1, f"{dark_web:,}", "Tarjetas en dark web", COLOR_FRAUD,
            f"{dark_web/total_cards*100:.1f}% del total" if total_cards else None)

        sin_chip = int((~cards["has_chip"]).sum()) if "has_chip" in cards.columns else 0
        kpi(c2, f"{sin_chip:,}", "Tarjetas sin chip", COLOR_WARN)

        if "credit_limit" in cards.columns:
            avg_limit = cards["credit_limit"].mean()
            kpi(c3, f"${avg_limit:,.0f}", "Límite crédito promedio", COLOR_OK)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            if "card_brand" in cards.columns:
                brand_counts = cards["card_brand"].value_counts().reset_index()
                brand_counts.columns = ["brand", "count"]
                fig_brand = px.pie(
                    brand_counts, names="brand", values="count",
                    color_discrete_sequence=[COLOR_OK, COLOR_USER, COLOR_FRAUD, COLOR_WARN],
                    hole=0.5, template=PLOTLY_TEMPLATE,
                    title="Distribución por marca"
                )
                fig_brand.update_layout(
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    height=300, margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_brand, use_container_width=True)

        with col2:
            if "card_type" in cards.columns and "card_on_dark_web" in cards.columns:
                risk_type = (cards.groupby("card_type")
                             .agg(total=("id", "count"),
                                  dark_web=("card_on_dark_web", "sum"))
                             .reset_index())
                risk_type["tasa_riesgo"] = risk_type["dark_web"] / risk_type["total"] * 100
                fig_risk = px.bar(
                    risk_type, x="card_type", y="tasa_riesgo",
                    color="tasa_riesgo",
                    color_continuous_scale=["#0e2a3f", COLOR_FRAUD],
                    labels={"card_type": "Tipo", "tasa_riesgo": "% en dark web"},
                    template=PLOTLY_TEMPLATE,
                    title="Riesgo por tipo de tarjeta",
                )
                fig_risk.update_layout(
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    height=300, margin=dict(l=0, r=0, t=40, b=0),
                    yaxis=dict(gridcolor="#1e2d47"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown('<div class="section-title">Perfil Demográfico de Usuarios</div>', unsafe_allow_html=True)

    if users is not None and len(users) > 0:
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
                    height=280, margin=dict(l=0, r=0, t=40, b=0),
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
                    color_continuous_scale=["#10b981", COLOR_WARN, COLOR_FRAUD],
                    size_max=8, opacity=0.6,
                    labels={"yearly_income": "Ingreso anual ($)",
                            "credit_score": "Credit Score",
                            "ratio_deuda": "Ratio deuda/ingreso"},
                    template=PLOTLY_TEMPLATE,
                    title="Ingreso vs Credit Score (color = ratio deuda)",
                )
                fig_scatter.update_layout(
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    height=280, margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Top usuarios con más fraudes
        if "client_id" in master.columns:
            st.markdown('<div class="section-title">Top 10 Usuarios con más Fraudes</div>',
                        unsafe_allow_html=True)
            top_users = (master[master["is_fraud"]]
                         .groupby("client_id")
                         .agg(fraudes=("id", "count"), monto_total=("amount", "sum"))
                         .reset_index()
                         .sort_values("fraudes", ascending=False)
                         .head(10))
            top_users = top_users.rename(columns={"client_id": "user_id"})

            if "credit_score" in users.columns:
                top_users = top_users.merge(
                    users[["id", "credit_score", "yearly_income"]].rename(columns={"id": "user_id"}),
                    on="user_id", how="left"
                )
            if not top_users.empty:
                st.dataframe(
                    top_users.style.background_gradient(
                        subset=["fraudes", "monto_total"], cmap="Reds"),
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
    elif len(master) == 0:
        st.info("No hay transacciones para analizar.")
    else:
        rng2 = np.random.default_rng(99)
        n = len(master)
        y_true = master["is_fraud"].astype(int).values

        fraud_scores = np.where(
            y_true == 1,
            rng2.beta(5, 2, n),
            rng2.beta(1, 6, n),
        )

        threshold = st.slider(
            "Umbral de decisión (fraud score)", 0.0, 1.0, 0.5, 0.01,
            help="Transacciones con score ≥ umbral se clasifican como fraude"
        )
        y_pred = (fraud_scores >= threshold).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        acc  = accuracy_score(y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, fraud_scores)
        roc_auc = auc(fpr, tpr)

        prec_curve, rec_curve, _ = precision_recall_curve(y_true, fraud_scores)
        pr_auc = auc(rec_curve, prec_curve)

        m1, m2, m3, m4, m5 = st.columns(5)
        kpi(m1, f"{prec:.3f}",    "Precision",  COLOR_OK)
        kpi(m2, f"{rec:.3f}",     "Recall",     COLOR_OK)
        kpi(m3, f"{f1:.3f}",      "F1-Score",   COLOR_OK)
        kpi(m4, f"{roc_auc:.3f}", "AUC-ROC",    COLOR_USER)
        kpi(m5, f"{pr_auc:.3f}",  "AUC-PR",     COLOR_USER)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"ROC (AUC={roc_auc:.3f})",
                line=dict(color=COLOR_USER, width=2.5)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random", line=dict(color=COLOR_MUTED, dash="dash", width=1)
            ))
            fig_roc.update_layout(
                title="Curva ROC", template=PLOTLY_TEMPLATE,
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=320, margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(title="FPR", gridcolor="#1e2d47"),
                yaxis=dict(title="TPR", gridcolor="#1e2d47"),
                legend=dict(x=0.6, y=0.1),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

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
                height=320, margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(title="Recall", gridcolor="#1e2d47"),
                yaxis=dict(title="Precision", gridcolor="#1e2d47"),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        with col3:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            fig_cm = go.Figure(go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Pred: Legítima", "Pred: Fraude"],
                y=["Real: Legítima", "Real: Fraude"],
                colorscale=[[0, "#0e2a3f"], [1, COLOR_FRAUD]],
                text=[[f"TN<br>{tn:,}", f"FP<br>{fp:,}"],
                      [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]],
                texttemplate="%{text}",
                textfont=dict(size=14, color="white"),
                showscale=False,
            ))
            fig_cm.update_layout(
                title="Matriz de Confusión", template=PLOTLY_TEMPLATE,
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=320, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

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
            line_color=COLOR_WARN, line_width=2,
            annotation_text=f"Umbral={threshold:.2f}",
            annotation_font_color=COLOR_WARN,
        )
        fig_score_dist.update_layout(
            barmode="overlay", template=PLOTLY_TEMPLATE,
            plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Fraud Score", yaxis_title="Cantidad",
            legend=dict(orientation="h", y=1.1),
            yaxis=dict(gridcolor="#1e2d47"),
        )
        st.plotly_chart(fig_score_dist, use_container_width=True)

        st.info(
            "💡 **Nota:** los fraud scores mostrados son simulados para visualización. "
            "Cuando entrenen el modelo, reemplazar `fraud_scores` con los valores reales."
        )


# ════════════════════════════════════════════════
#  TAB 5 — ANÁLISIS POR USUARIO  (NUEVO)
# ════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Vista general de la base de usuarios</div>',
                unsafe_allow_html=True)

    # Cálculo de métricas agregadas por usuario
    if "client_id" in master.columns and len(master) > 0:
        gasto_x_user = (master.groupby("client_id")
                        .agg(cant_transacciones=("id", "count"),
                             gasto_total=("amount", "sum"),
                             gasto_promedio=("amount", "mean"),
                             fraudes=("is_fraud", "sum"))
                        .reset_index()
                        .rename(columns={"client_id": "id"}))

        if cards is not None and "client_id" in cards.columns:
            n_cards = (cards.groupby("client_id").size()
                       .reset_index(name="num_tarjetas")
                       .rename(columns={"client_id": "id"}))
        else:
            n_cards = pd.DataFrame(columns=["id", "num_tarjetas"])

        cols_u = ["id"]
        for c in ["current_age", "gender", "yearly_income", "credit_score",
                  "total_debt", "latitude", "longitude", "address"]:
            if c in users.columns:
                cols_u.append(c)

        perfil = users[cols_u].merge(gasto_x_user, on="id", how="left")
        perfil = perfil.merge(n_cards, on="id", how="left")
        for col in ["cant_transacciones", "gasto_total", "gasto_promedio",
                    "fraudes", "num_tarjetas"]:
            if col in perfil.columns:
                perfil[col] = perfil[col].fillna(0)
    else:
        perfil = pd.DataFrame()

    if perfil.empty:
        st.warning("⚠️  No hay datos suficientes para mostrar análisis por usuario.")
    else:
        # KPIs generales
        total_users = len(perfil)
        users_activos = int((perfil["cant_transacciones"] > 0).sum())
        avg_gasto = perfil.loc[perfil["cant_transacciones"] > 0, "gasto_total"].mean()
        avg_tarjetas = perfil["num_tarjetas"].mean() if "num_tarjetas" in perfil.columns else 0
        avg_edad = perfil["current_age"].mean() if "current_age" in perfil.columns else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        kpi(k1, f"{total_users:,}",       "Usuarios totales",     COLOR_OK)
        kpi(k2, f"{users_activos:,}",     "Con transacciones",    COLOR_OK,
            f"{users_activos/total_users*100:.1f}% del total" if total_users else None)
        kpi(k3, f"${avg_gasto:,.0f}",     "Gasto promedio",       COLOR_USER)
        kpi(k4, f"{avg_tarjetas:.1f}",    "Tarjetas promedio",    COLOR_WARN)
        kpi(k5, f"{avg_edad:.0f} años",   "Edad promedio",        COLOR_OK)

        st.markdown("<br>", unsafe_allow_html=True)

        # Top usuarios
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("**Top 15 usuarios por gasto total**")
            top_gasto = perfil.nlargest(15, "gasto_total")[
                ["id", "current_age", "gender", "gasto_total",
                 "cant_transacciones", "num_tarjetas"]
            ].copy()
            top_gasto.columns = ["ID", "Edad", "Género", "Gasto total ($)",
                                 "Transacciones", "Tarjetas"]
            st.dataframe(
                top_gasto.style.background_gradient(
                    subset=["Gasto total ($)"], cmap="Oranges"
                ).format({"Gasto total ($)": "${:,.0f}",
                          "Transacciones": "{:,.0f}",
                          "Tarjetas": "{:,.0f}"}),
                use_container_width=True, hide_index=True, height=420
            )

        with col_b:
            st.markdown("**Distribución del gasto entre usuarios**")
            if (perfil["gasto_total"] > 0).any():
                p99 = perfil["gasto_total"].quantile(0.99)
                fig_dist = go.Figure(go.Histogram(
                    x=perfil.loc[perfil["gasto_total"] > 0, "gasto_total"].clip(upper=p99),
                    nbinsx=40,
                    marker_color=COLOR_USER, opacity=0.85,
                ))
                fig_dist.update_layout(
                    template=PLOTLY_TEMPLATE,
                    plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                    height=420, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(title="Gasto total ($)", gridcolor="#1e2d47"),
                    yaxis=dict(title="Cantidad de usuarios", gridcolor="#1e2d47"),
                    showlegend=False,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        # Selector de usuario individual
        st.markdown("---")
        st.markdown('<div class="section-title">🔎 Análisis individual de usuario</div>',
                    unsafe_allow_html=True)

        user_ids = sorted(perfil["id"].unique().tolist())
        default_user = perfil.nlargest(1, "gasto_total")["id"].iloc[0]
        try:
            default_idx = user_ids.index(default_user)
        except ValueError:
            default_idx = 0

        col_sel1, col_sel2 = st.columns([1, 3])
        with col_sel1:
            user_sel = st.selectbox(
                "Seleccioná un usuario",
                options=user_ids,
                index=default_idx,
            )
        with col_sel2:
            st.markdown(
                f"<div style='padding-top:32px;color:{COLOR_MUTED};font-size:0.85rem'>"
                f"💡 Tip: para ver usuarios interesantes, fijate en la tabla de arriba "
                f"y usá esos IDs en el selector.</div>",
                unsafe_allow_html=True
            )

        user_row = perfil[perfil["id"] == user_sel].iloc[0]
        user_txn = master[master["client_id"] == user_sel]

        # KPIs del usuario individual
        st.markdown("<br>", unsafe_allow_html=True)
        u1, u2, u3, u4, u5, u6 = st.columns(6)
        kpi(u1,
            f"{int(user_row.get('current_age', 0))} años" if pd.notna(user_row.get('current_age')) else "—",
            "Edad", COLOR_OK)
        kpi(u2,
            f"${user_row.get('yearly_income', 0):,.0f}" if pd.notna(user_row.get('yearly_income')) else "—",
            "Ingreso anual", COLOR_OK)
        kpi(u3, f"{int(user_row.get('num_tarjetas', 0))}", "Tarjetas", COLOR_USER)
        kpi(u4, f"{int(user_row.get('cant_transacciones', 0)):,}", "Transacciones", COLOR_OK)
        kpi(u5, f"${user_row.get('gasto_total', 0):,.0f}", "Gasto total", COLOR_USER)

        score = user_row.get("credit_score")
        if pd.notna(score):
            score_color = COLOR_OK if score >= 700 else (COLOR_WARN if score >= 600 else COLOR_FRAUD)
            kpi(u6, f"{int(score)}", "Credit Score", score_color)
        else:
            kpi(u6, "—", "Credit Score", COLOR_MUTED)

        # Mapa
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">🗺️  Mapa de actividad — Usuario {user_sel}</div>',
                    unsafe_allow_html=True)

        user_lat = user_row.get("latitude")
        user_lon = user_row.get("longitude")
        user_dir = user_row.get("address", "Dirección no disponible")
        tiene_geo_user = pd.notna(user_lat) and pd.notna(user_lon)
        tiene_geo_txn = (not user_txn.empty and
                         user_txn[["merchant_lat", "merchant_lon"]].notna().any().any())

        if not tiene_geo_user and not tiene_geo_txn:
            st.info("ℹ️  Este usuario no tiene coordenadas geográficas disponibles.")
        else:
            fig_mapa = go.Figure()

            if tiene_geo_txn:
                tx_geo = user_txn.dropna(subset=["merchant_lat", "merchant_lon"]).copy()
                if not tx_geo.empty:
                    tx_geo["size"] = np.clip(np.sqrt(tx_geo["amount"]) * 1.5, 4, 25)
                    tx_geo["color"] = tx_geo["is_fraud"].map(
                        {True: COLOR_FRAUD, False: COLOR_OK}
                    )

                    fig_mapa.add_trace(go.Scattermapbox(
                        lat=tx_geo["merchant_lat"],
                        lon=tx_geo["merchant_lon"],
                        mode="markers",
                        marker=dict(size=tx_geo["size"], color=tx_geo["color"], opacity=0.7),
                        text=tx_geo.apply(
                            lambda r: (f"<b>${r['amount']:,.2f}</b><br>"
                                       f"{r.get('merchant_city', '')}, "
                                       f"{r.get('merchant_state', '')}<br>"
                                       f"{'⚠️ FRAUDE' if r['is_fraud'] else '✓ Legítima'}"),
                            axis=1
                        ),
                        hovertemplate="%{text}<extra></extra>",
                        name="Transacciones",
                    ))

            if tiene_geo_user:
                fig_mapa.add_trace(go.Scattermapbox(
                    lat=[user_lat], lon=[user_lon],
                    mode="markers",
                    marker=dict(size=22, color=COLOR_USER),
                    text=[f"<b>🏠 Usuario {user_sel}</b><br>{user_dir}"],
                    hovertemplate="%{text}<extra></extra>",
                    name=f"Domicilio Usuario {user_sel}",
                ))

            if tiene_geo_user:
                center_lat, center_lon = user_lat, user_lon
            elif tiene_geo_txn and not tx_geo.empty:
                center_lat = tx_geo["merchant_lat"].mean()
                center_lon = tx_geo["merchant_lon"].mean()
            else:
                center_lat, center_lon = 39.5, -98.35

            fig_mapa.update_layout(
                mapbox=dict(
                    style="carto-darkmatter",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=4,
                ),
                template=PLOTLY_TEMPLATE,
                plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                height=480, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                            bgcolor="rgba(17,24,39,0.8)"),
            )
            st.plotly_chart(fig_mapa, use_container_width=True)

            leyenda = []
            if tiene_geo_user:
                leyenda.append("🟣 Domicilio del usuario")
            if tiene_geo_txn:
                leyenda.append("🟦 Transacciones legítimas (tamaño = monto)")
                if user_txn["is_fraud"].any():
                    leyenda.append("🟧 Transacciones fraudulentas")
            leyenda.append("ℹ️ Las transacciones se ubican en el centro de cada estado (aproximación)")
            st.caption(" · ".join(leyenda))

        # Patrones de consumo
        if not user_txn.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Patrones de consumo del usuario</div>',
                        unsafe_allow_html=True)

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.markdown("**Evolución mensual de gasto**")
                if "month" in user_txn.columns:
                    evol = (user_txn.groupby("month")
                            .agg(monto=("amount", "sum"))
                            .reset_index()
                            .sort_values("month"))
                    fig_evol = go.Figure()
                    fig_evol.add_trace(go.Bar(
                        x=evol["month"], y=evol["monto"],
                        marker_color=COLOR_USER, opacity=0.85,
                    ))
                    fig_evol.update_layout(
                        template=PLOTLY_TEMPLATE,
                        plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                        height=300, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(title="Mes", gridcolor="#1e2d47"),
                        yaxis=dict(title="Monto ($)", gridcolor="#1e2d47"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_evol, use_container_width=True)

            with col_g2:
                st.markdown("**Top categorías de consumo**")
                if "mcc_desc" in user_txn.columns:
                    top_mcc = (user_txn.groupby("mcc_desc")
                               .agg(monto=("amount", "sum"))
                               .reset_index()
                               .sort_values("monto", ascending=False)
                               .head(10))
                    fig_mcc_user = px.bar(
                        top_mcc.sort_values("monto"),
                        x="monto", y="mcc_desc", orientation="h",
                        color="monto",
                        color_continuous_scale=["#1e3a5f", COLOR_USER],
                        labels={"monto": "Monto ($)", "mcc_desc": "Categoría"},
                        template=PLOTLY_TEMPLATE,
                    )
                    fig_mcc_user.update_layout(
                        plot_bgcolor=COLOR_BG, paper_bgcolor=COLOR_BG,
                        height=300, margin=dict(l=0, r=0, t=10, b=0),
                        yaxis=dict(tickfont=dict(size=10)),
                        xaxis=dict(gridcolor="#1e2d47"),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_mcc_user, use_container_width=True)

            # Últimas transacciones del usuario
            st.markdown("**Últimas 20 transacciones**")
            cols = [c for c in ["date", "amount", "mcc_desc", "merchant_city",
                                "merchant_state", "is_fraud"]
                    if c in user_txn.columns]
            ultimas = user_txn.sort_values("date", ascending=False).head(20)[cols].copy()
            if "amount" in ultimas.columns:
                ultimas["amount"] = ultimas["amount"].map("${:,.2f}".format)
            if "is_fraud" in ultimas.columns:
                ultimas["is_fraud"] = ultimas["is_fraud"].map({True: "⚠️ FRAUDE", False: "✓"})
            st.dataframe(ultimas, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════
#  TAB 6 — CARGAR TRANSACCIÓN  (NUEVO — TIEMPO REAL)
# ════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">➕ Agregar transacción nueva (tiempo real)</div>',
                unsafe_allow_html=True)

    st.markdown(
        f"<div style='color:{COLOR_MUTED};font-size:0.9rem;margin-bottom:20px'>"
        "Esta vista demuestra el procesamiento incremental: al agregar una "
        "transacción, los KPIs y métricas del resto del dashboard se actualizan "
        "<b>al instante</b>, sin reprocesar el CSV ni reiniciar la aplicación."
        "</div>",
        unsafe_allow_html=True
    )

    # Mostrar el último ID para sugerir uno nuevo
    next_id = int(master_full["id"].max()) + 1 if len(master_full) > 0 else 99000000

    with st.form("nueva_transaccion", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            tx_id = st.number_input("ID de transacción", value=next_id, step=1, format="%d")
            client_id = st.number_input("ID de usuario (client_id)",
                                         value=int(users["id"].iloc[0]) if len(users) > 0 else 0,
                                         step=1, format="%d")
            # Tarjetas del cliente seleccionado
            tarjetas_cliente = cards[cards["client_id"] == client_id]["id"].tolist() if cards is not None else []
            if tarjetas_cliente:
                card_id = st.selectbox("Tarjeta", options=tarjetas_cliente)
            else:
                card_id = st.number_input("ID de tarjeta", value=0, step=1, format="%d")
                st.caption(f"⚠️ El usuario {client_id} no tiene tarjetas en la base.")

        with col2:
            amount = st.number_input("Monto ($)", min_value=0.01, value=100.0, step=10.0, format="%.2f")
            fecha = st.date_input("Fecha", value=date.today())
            hora = st.time_input("Hora", value=datetime.now().time())
            use_chip = st.selectbox("Tipo de uso",
                                     ["Chip Transaction", "Swipe Transaction", "Online Transaction"])

        with col3:
            merchant_id = st.number_input("ID de comercio", value=99999, step=1, format="%d")
            merchant_city = st.text_input("Ciudad del comercio", value="Buenos Aires")
            estados_validos = sorted(master_full["merchant_state"].dropna().unique().tolist())
            merchant_state = st.selectbox("Estado", options=estados_validos,
                                           index=estados_validos.index("CA") if "CA" in estados_validos else 0)
            mcc_options = master_full["mcc"].dropna().unique().tolist()
            mcc = st.selectbox("Código MCC", options=sorted(mcc_options) if mcc_options else ["5812"])

        submitted = st.form_submit_button("➕ Agregar transacción",
                                           type="primary", use_container_width=True)

        if submitted:
            if amount <= 0:
                st.error("El monto debe ser mayor a cero.")
            elif client_id not in users["id"].values:
                st.error(f"El usuario {client_id} no existe en la base.")
            else:
                # Buscar coordenadas aproximadas del estado
                from carga_inicial import STATE_COORDS  # reutilizar la misma tabla
                coords = STATE_COORDS.get(merchant_state, (None, None))

                fecha_completa = datetime.combine(fecha, hora)
                tx_dict = {
                    "transaction_id": int(tx_id),
                    "date": fecha_completa.strftime("%Y-%m-%d %H:%M:%S"),
                    "client_id": int(client_id),
                    "card_id": int(card_id),
                    "amount": float(amount),
                    "use_chip": use_chip,
                    "merchant_id": int(merchant_id),
                    "merchant_city": merchant_city,
                    "merchant_state": merchant_state,
                    "merchant_lat": coords[0],
                    "merchant_lon": coords[1],
                    "zip": None,
                    "mcc": str(mcc),
                    "year": fecha_completa.year,
                    "month": fecha_completa.strftime("%Y-%m"),
                }

                resultado = agregar_transaccion_db(tx_dict)
                if resultado["ok"]:
                    st.success(resultado["mensaje"])
                    # Invalidar el cache para que las próximas vistas vean el dato nuevo
                    st.session_state.cache_key += 1
                    st.cache_data.clear()
                    st.info("🔄 Refrescá la página o cambiá de tab para ver los KPIs actualizados.")
                else:
                    st.error(resultado["mensaje"])

    # Métricas en vivo del último estado de la base
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Estado actual de la base</div>',
                unsafe_allow_html=True)

    conn_check = _conn()
    estado = pd.read_sql_query("SELECT * FROM agg_global", conn_check).iloc[0]
    conn_check.close()

    e1, e2, e3, e4 = st.columns(4)
    kpi(e1, f"{int(estado['cant_transacciones']):,}",  "Transacciones totales", COLOR_OK)
    kpi(e2, f"${estado['monto_total']:,.0f}",          "Monto total operado",   COLOR_OK)
    kpi(e3, f"${estado['monto_promedio']:,.2f}",       "Monto promedio",        COLOR_USER)
    kpi(e4, f"{int(estado['cant_usuarios_activos']):,}", "Usuarios activos",     COLOR_OK)


# ════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════
st.markdown("""
<hr style='border-color:#1e2d47;margin-top:40px'>
<p style='text-align:center;color:#334155;font-size:11px'>
  FinTrack v2 · Dashboard con SQLite + procesamiento incremental ·
  Ingeniería de Software · Ciencia de Datos en Organizaciones · 2026
</p>
""", unsafe_allow_html=True)
