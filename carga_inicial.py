"""
================================================================================
  FinTrack — Carga Inicial y Limpieza de Datos (v2)
================================================================================

Igual que la versión 1, pero agrega:
  - Columnas derivadas (year, month) para facilitar agregaciones temporales.
  - Tabla `mcc_codes` enriquecida con `grupo` (Retail, Hotelería, etc.) que
    usa el dashboard.
  - Coordenadas aproximadas por estado (merchant_lat, merchant_lon) para
    poder dibujar las transacciones en el mapa.

Uso:
    python carga_inicial.py
================================================================================
"""

import pandas as pd
import sqlite3
import json
import re
from pathlib import Path

# ============================================================================
#   CONFIGURACIÓN
# ============================================================================

DATA_DIR = Path("./data")
DB_PATH  = Path("./fintrack.db")

ARCHIVO_USERS         = DATA_DIR / "users_data.csv"
ARCHIVO_CARDS         = DATA_DIR / "cards_data.csv"
ARCHIVO_TRANSACTIONS  = DATA_DIR / "transactions_data.csv"
ARCHIVO_MCC           = DATA_DIR / "mcc_codes.json"
ARCHIVO_FRAUD_LABELS  = DATA_DIR / "train_fraud_labels.json"


# ============================================================================
#   AGRUPACIÓN DE CÓDIGOS MCC EN CATEGORÍAS GRANDES
#   (rangos estándar de la industria de tarjetas)
# ============================================================================

def _mcc_group(code):
    """Asigna un grupo descriptivo a un código MCC según su rango."""
    try:
        c = int(code)
    except (ValueError, TypeError):
        return "Otro"
    if 700  <= c <= 1499:  return "Agricultura"
    if 1500 <= c <= 2999:  return "Transporte y Servicios Públicos"
    if 3000 <= c <= 3999:  return "Aerolíneas y Hoteles"
    if 4000 <= c <= 4799:  return "Transporte"
    if 4800 <= c <= 4999:  return "Servicios Públicos / Telecom"
    if 5000 <= c <= 5599:  return "Retail / Comercio"
    if 5600 <= c <= 5699:  return "Indumentaria"
    if 5700 <= c <= 7299:  return "Otros minoristas y servicios"
    if 7300 <= c <= 7999:  return "Servicios y entretenimiento"
    if 8000 <= c <= 8999:  return "Salud y Educación"
    if 9000 <= c <= 9999:  return "Gobierno"
    return "Otro"


# ============================================================================
#   COORDENADAS APROXIMADAS DE LOS ESTADOS DE EE.UU.
#   (centro geográfico de cada estado — para dibujar transacciones en el mapa)
# ============================================================================

STATE_COORDS = {
    "AL": (32.806, -86.791),  "AK": (61.370, -152.404), "AZ": (33.730, -111.431),
    "AR": (34.969, -92.373),  "CA": (36.116, -119.682), "CO": (39.059, -105.311),
    "CT": (41.597, -72.755),  "DE": (39.318, -75.507),  "FL": (27.766, -81.687),
    "GA": (33.040, -83.643),  "HI": (21.094, -157.498), "ID": (44.240, -114.479),
    "IL": (40.349, -88.986),  "IN": (39.849, -86.258),  "IA": (42.011, -93.210),
    "KS": (38.526, -96.726),  "KY": (37.668, -84.670),  "LA": (31.169, -91.867),
    "ME": (44.693, -69.381),  "MD": (39.063, -76.802),  "MA": (42.230, -71.530),
    "MI": (43.326, -84.536),  "MN": (45.694, -93.900),  "MS": (32.741, -89.678),
    "MO": (38.456, -92.288),  "MT": (46.921, -110.454), "NE": (41.125, -98.268),
    "NV": (38.313, -117.055), "NH": (43.452, -71.563),  "NJ": (40.298, -74.521),
    "NM": (34.840, -106.248), "NY": (42.165, -74.948),  "NC": (35.630, -79.806),
    "ND": (47.528, -99.784),  "OH": (40.388, -82.764),  "OK": (35.565, -96.928),
    "OR": (44.572, -122.070), "PA": (40.590, -77.209),  "RI": (41.680, -71.511),
    "SC": (33.856, -80.945),  "SD": (44.299, -99.438),  "TN": (35.747, -86.692),
    "TX": (31.054, -97.563),  "UT": (40.150, -111.862), "VT": (44.045, -72.710),
    "VA": (37.769, -78.169),  "WA": (47.400, -121.490), "WV": (38.491, -80.954),
    "WI": (44.268, -89.616),  "WY": (42.756, -107.302), "DC": (38.897, -77.026),
}


# ============================================================================
#   FUNCIONES DE LIMPIEZA
# ============================================================================

def limpiar_monto(valor):
    """Convierte '$1,234.56', '$-77.00', '$0' a float. None si no se puede."""
    if pd.isna(valor):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    limpio = re.sub(r"[\$,\s]", "", str(valor))
    try:
        return float(limpio)
    except ValueError:
        return None


def cargar_y_limpiar_usuarios():
    print("\n[1/3] Procesando usuarios...")
    df = pd.read_csv(ARCHIVO_USERS)
    n_inicial = len(df)

    df = df.rename(columns={"id": "user_id"})

    for col in ["per_capita_income", "yearly_income", "total_debt"]:
        df[col] = df[col].apply(limpiar_monto)

    df = df[df["user_id"].notna()]
    df = df[(df["current_age"] >= 18) & (df["current_age"] <= 110)]
    df = df[df["yearly_income"].notna() & (df["yearly_income"] > 0)]

    mask_geo_invalida = (df["latitude"] == 0) & (df["longitude"] == 0)
    df.loc[mask_geo_invalida, ["latitude", "longitude"]] = None

    print(f"    Usuarios cargados: {len(df):,} (descartados: {n_inicial - len(df):,})")
    return df


def cargar_y_limpiar_tarjetas():
    print("\n[2/3] Procesando tarjetas...")
    df = pd.read_csv(ARCHIVO_CARDS)
    n_inicial = len(df)

    df = df.rename(columns={"id": "card_id"})
    df["credit_limit"] = df["credit_limit"].apply(limpiar_monto)
    df["has_chip"]         = df["has_chip"].map({"YES": 1, "NO": 0})
    df["card_on_dark_web"] = df["card_on_dark_web"].map({"Yes": 1, "No": 0})

    df = df[df["card_id"].notna() & df["client_id"].notna()]
    df = df[df["credit_limit"].notna() & (df["credit_limit"] > 0)]

    df = df.drop(columns=["card_number", "cvv", "year_pin_last_changed"], errors="ignore")

    print(f"    Tarjetas cargadas: {len(df):,} (descartadas: {n_inicial - len(df):,})")
    return df


def cargar_y_limpiar_transacciones():
    print("\n[3/3] Procesando transacciones (puede tardar un poco)...")
    df = pd.read_csv(ARCHIVO_TRANSACTIONS, low_memory=False)
    n_inicial = len(df)

    df = df.rename(columns={"id": "transaction_id"})

    df["amount"] = df["amount"].apply(limpiar_monto)
    df = df[df["amount"].notna() & (df["amount"] > 0)]
    df = df[df["errors"].isna() | (df["errors"].astype(str).str.strip() == "")]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]

    # Columnas derivadas para el dashboard
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)

    df["zip"] = df["zip"].apply(lambda x: str(int(x)) if pd.notna(x) else None)
    df["mcc"] = df["mcc"].astype(str).str.replace(".0", "", regex=False)

    # Coordenadas aproximadas por estado
    df["merchant_lat"] = df["merchant_state"].map(
        lambda s: STATE_COORDS.get(s, (None, None))[0]
    )
    df["merchant_lon"] = df["merchant_state"].map(
        lambda s: STATE_COORDS.get(s, (None, None))[1]
    )

    df = df.drop(columns=["errors"])

    print(f"    Transacciones cargadas: {len(df):,} (descartadas: {n_inicial - len(df):,})")
    return df


def cargar_mcc_codes():
    """Carga MCC y agrega columna de grupo."""
    with open(ARCHIVO_MCC, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(list(data.items()), columns=["mcc", "descripcion"])
    df["grupo"] = df["mcc"].apply(_mcc_group)
    return df


def cargar_fraud_labels():
    if not ARCHIVO_FRAUD_LABELS.exists():
        return None
    with open(ARCHIVO_FRAUD_LABELS, "r", encoding="utf-8") as f:
        data = json.load(f)
    target = data.get("target", data)
    df = pd.DataFrame(
        [(int(k), 1 if v == "Yes" else 0) for k, v in target.items()],
        columns=["transaction_id", "is_fraud"]
    )
    return df


# ============================================================================
#   CONSTRUCCIÓN DE LA BASE
# ============================================================================

def construir_base(users_df, cards_df, tx_df, mcc_df, fraud_df):
    print(f"\n→ Creando base de datos en: {DB_PATH.resolve()}")

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    users_df.to_sql("users",        conn, index=False)
    cards_df.to_sql("cards",        conn, index=False)
    tx_df.to_sql("transactions",    conn, index=False)
    mcc_df.to_sql("mcc_codes",      conn, index=False)
    if fraud_df is not None:
        fraud_df.to_sql("fraud_labels", conn, index=False)

    cur = conn.cursor()

    cur.executescript("""
        CREATE INDEX idx_tx_client    ON transactions(client_id);
        CREATE INDEX idx_tx_date      ON transactions(date);
        CREATE INDEX idx_tx_merchant  ON transactions(merchant_id);
        CREATE INDEX idx_tx_state     ON transactions(merchant_state);
        CREATE INDEX idx_cards_client ON cards(client_id);
    """)

    print("→ Pre-calculando agregados por usuario...")
    cur.execute("""
        CREATE TABLE agg_por_usuario AS
        SELECT
            u.user_id,
            u.current_age,
            u.gender,
            u.yearly_income,
            u.credit_score,
            u.num_credit_cards,
            u.latitude,
            u.longitude,
            u.address,
            COUNT(t.transaction_id)        AS cant_transacciones,
            COALESCE(SUM(t.amount), 0)     AS gasto_total,
            COALESCE(AVG(t.amount), 0)     AS gasto_promedio,
            MIN(t.amount)                  AS gasto_minimo,
            MAX(t.amount)                  AS gasto_maximo,
            MIN(t.date)                    AS primera_transaccion,
            MAX(t.date)                    AS ultima_transaccion
        FROM users u
        LEFT JOIN transactions t ON t.client_id = u.user_id
        GROUP BY u.user_id;
    """)
    cur.execute("CREATE UNIQUE INDEX idx_agg_user ON agg_por_usuario(user_id);")

    print("→ Pre-calculando agregados por comercio...")
    cur.execute("""
        CREATE TABLE agg_por_comercio AS
        SELECT
            merchant_id,
            MAX(merchant_city)   AS merchant_city,
            MAX(merchant_state)  AS merchant_state,
            COUNT(*)             AS cant_transacciones,
            SUM(amount)          AS monto_total,
            AVG(amount)          AS monto_promedio
        FROM transactions
        GROUP BY merchant_id;
    """)
    cur.execute("CREATE UNIQUE INDEX idx_agg_merchant ON agg_por_comercio(merchant_id);")

    print("→ Pre-calculando agregado global...")
    cur.execute("""
        CREATE TABLE agg_global AS
        SELECT
            COUNT(*)                    AS cant_transacciones,
            SUM(amount)                 AS monto_total,
            AVG(amount)                 AS monto_promedio,
            COUNT(DISTINCT client_id)   AS cant_usuarios_activos
        FROM transactions;
    """)

    conn.commit()
    conn.close()
    print("✓ Base creada exitosamente.\n")


def main():
    print("=" * 70)
    print("  FinTrack — Carga inicial de datos (v2)")
    print("=" * 70)

    users_df  = cargar_y_limpiar_usuarios()
    cards_df  = cargar_y_limpiar_tarjetas()
    tx_df     = cargar_y_limpiar_transacciones()
    mcc_df    = cargar_mcc_codes()
    fraud_df  = cargar_fraud_labels()

    construir_base(users_df, cards_df, tx_df, mcc_df, fraud_df)

    print("=" * 70)
    print("  Resumen")
    print("=" * 70)
    print(f"  Usuarios:       {len(users_df):>10,}")
    print(f"  Tarjetas:       {len(cards_df):>10,}")
    print(f"  Transacciones:  {len(tx_df):>10,}")
    print(f"  Códigos MCC:    {len(mcc_df):>10,}")
    if fraud_df is not None:
        print(f"  Etiquetas fraude: {len(fraud_df):>8,}")
    print(f"\n  Base de datos generada en: {DB_PATH.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
