"""
===============================================================
  PREPARACIÓN DE DATOS — Sistema de Detección de Fraude
  Ingeniería de Software · Ciencia de Datos en Organizaciones
===============================================================
Script unificado para:
  1. Inspeccionar cada archivo (tamaño, shape, nulos, duplicados)
  2. Limpiar filas con errores o inconsistencias
  3. Comprimir / convertir a formato eficiente (Parquet o CSV.GZ)
  4. Validar el resultado final

Requisitos:
  pip install pandas pyarrow tqdm

Uso:
  python preparar_datos.py

Los archivos limpios se guardan en la carpeta  ./data/clean/
===============================================================
"""

import os
import json
import time
import gzip
import shutil
from pathlib import Path

import pandas as pd

# ── opcional: barra de progreso
try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

# ── opcional: Parquet (mucho más eficiente que CSV para análisis)
try:
    import pyarrow          # noqa
    PARQUET = True
except ImportError:
    PARQUET = False
    print("⚠️  pyarrow no instalado → se usará CSV.GZ en lugar de Parquet")
    print("   Para instalar: pip install pyarrow\n")

# ─────────────────────────────────────────────
#  RUTAS  (ajustá si descargaste en otro lugar)
# ─────────────────────────────────────────────
RAW_DIR   = Path("./data/raw")       # donde están los archivos originales
CLEAN_DIR = Path("./data/clean")     # donde se guardan los archivos limpios
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

TRANSACTIONS_CSV   = RAW_DIR / "transactions_data.csv"
CARDS_CSV          = RAW_DIR / "cards_data.csv"
USERS_CSV          = RAW_DIR / "users_data.csv"
FRAUD_LABELS_JSON  = RAW_DIR / "train_fraud_labels.json"
MCC_CODES_JSON     = RAW_DIR / "mcc_codes.json"


# ══════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════

def separador(titulo: str):
    print("\n" + "═" * 60)
    print(f"  {titulo}")
    print("═" * 60)

def reporte_basico(df: pd.DataFrame, nombre: str):
    """Imprime estadísticas rápidas de un DataFrame."""
    print(f"\n📊 {nombre}")
    print(f"   Filas      : {df.shape[0]:,}")
    print(f"   Columnas   : {df.shape[1]}")
    print(f"   Mem. uso   : {df.memory_usage(deep=True).sum() / 1_048_576:.1f} MB")
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if len(nulos):
        print(f"   Nulos por columna:")
        for col, n in nulos.items():
            print(f"     • {col}: {n:,} ({n/len(df)*100:.1f}%)")
    else:
        print("   Nulos      : ninguno ✓")
    dupes = df.duplicated().sum()
    print(f"   Duplicados : {dupes:,}")

def guardar(df: pd.DataFrame, nombre_base: str):
    """Guarda en Parquet si está disponible, sino en CSV comprimido."""
    if PARQUET:
        out = CLEAN_DIR / f"{nombre_base}.parquet"
        df.to_parquet(out, index=False, compression="snappy")
        print(f"   💾 Guardado → {out}  ({out.stat().st_size / 1_048_576:.1f} MB)")
    else:
        out = CLEAN_DIR / f"{nombre_base}.csv.gz"
        df.to_csv(out, index=False, compression="gzip")
        print(f"   💾 Guardado → {out}  ({out.stat().st_size / 1_048_576:.1f} MB)")
    return out

def tamaño_mb(path: Path) -> float:
    return path.stat().st_size / 1_048_576 if path.exists() else 0.0


# ══════════════════════════════════════════════
#  1. TRANSACTIONS_DATA.CSV  (~1.2 GB)
# ══════════════════════════════════════════════

def limpiar_transacciones():
    separador("1 · transactions_data.csv")
    print(f"   Tamaño original: {tamaño_mb(TRANSACTIONS_CSV):.0f} MB")
    print("   Leyendo en chunks (archivo grande)…")
    t0 = time.time()

    # Tipos explícitos para reducir memoria desde la lectura.
    # NOTA: NO incluimos "amount" aquí para leerlo siempre como string
    # y convertirlo nosotros mismos de forma segura (evita conflictos
    # con pandas 3.x + ArrowDtype cuando el valor tiene "$" o ",").
    dtype_map = {
        "id"             : "int32",
        "card_id"        : "int32",
        "client_id"      : "int32",
        "mcc"            : "str",
        "use_chip"       : "str",
        "merchant_city"  : "str",
        "merchant_state" : "str",
        "errors"         : "str",
        "amount"         : "str",   # lo convertimos manualmente abajo
    }

    chunks = []
    filas_originales  = 0
    filas_descartadas = 0

    # Leemos en chunks de 500k filas para no saturar RAM
    CHUNK_SIZE = 500_000
    reader = pd.read_csv(
        TRANSACTIONS_CSV,
        dtype=dtype_map,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        filas_originales += len(chunk)

        # ── Limpieza ──────────────────────────────────────

        # 1. Eliminar filas completamente vacías
        chunk.dropna(how="all", inplace=True)

        # 2. El monto debe ser numérico y positivo.
        #    Leído como str para manejar "$1,234.56".
        #    Forzamos conversión explícita compatible con pandas 3.x + ArrowDtype.
        chunk["amount"] = (
            chunk["amount"]
            .astype(str)
            .str.replace(r"[\$,\s]", "", regex=True)
            .replace({"nan": None, "": None})
        )
        chunk["amount"] = pd.to_numeric(chunk["amount"], errors="coerce")
        # Descartar montos nulos o negativos usando numpy para evitar
        # problemas de comparación con ArrowDtype
        antes = len(chunk)
        amount_vals = pd.to_numeric(chunk["amount"], errors="coerce")
        chunk = chunk[amount_vals.notna() & (amount_vals >= 0)]
        filas_descartadas += antes - len(chunk)

        # 3. Convertir datetime con formato estándar
        if "date" in chunk.columns:
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            # Descartar fechas imposibles
            antes = len(chunk)
            chunk = chunk[chunk["date"].notna()]
            filas_descartadas += antes - len(chunk)

        # 4. Normalizar columna errors: vacío → "None"
        if "errors" in chunk.columns:
            chunk["errors"] = chunk["errors"].fillna("None")

        # 5. Eliminar duplicados dentro del chunk por id de transacción
        if "id" in chunk.columns:
            chunk.drop_duplicates(subset=["id"], keep="first", inplace=True)

        chunks.append(chunk)

        if (i + 1) % 5 == 0:
            print(f"   … {filas_originales:,} filas procesadas")

    df = pd.concat(chunks, ignore_index=True)

    # 6. Deduplicación global (entre chunks)
    if "id" in df.columns:
        antes = len(df)
        df.drop_duplicates(subset=["id"], keep="first", inplace=True)
        filas_descartadas += antes - len(df)

    print(f"\n   ✅ Filas originales  : {filas_originales:,}")
    print(f"   🗑️  Filas descartadas : {filas_descartadas:,}  ({filas_descartadas/filas_originales*100:.2f}%)")
    print(f"   ✓  Filas limpias     : {len(df):,}")
    print(f"   ⏱️  Tiempo            : {time.time()-t0:.1f}s")

    reporte_basico(df, "transactions (limpio)")
    out = guardar(df, "transactions_clean")

    # Comparación de tamaño
    orig_mb  = tamaño_mb(TRANSACTIONS_CSV)
    clean_mb = tamaño_mb(out)
    print(f"   📉 Compresión: {orig_mb:.0f} MB → {clean_mb:.0f} MB  "
          f"({(1 - clean_mb/orig_mb)*100:.0f}% reducción)")
    return df


# ══════════════════════════════════════════════
#  2. CARDS_DATA.CSV  (~500 KB)
# ══════════════════════════════════════════════

def limpiar_tarjetas():
    separador("2 · cards_data.csv")
    df = pd.read_csv(CARDS_CSV, low_memory=False)
    reporte_basico(df, "cards (original)")

    # ── Limpieza ──────────────────────────────────────

    # 1. credit_limit viene como "$24,295" → numérico
    # Forzar conversión siempre (compatible con pandas 3.x + ArrowDtype)
    df["credit_limit"] = (
        df["credit_limit"]
        .astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)
        .replace({"nan": None, "": None})
    )
    df["credit_limit"] = pd.to_numeric(df["credit_limit"], errors="coerce")

    # 2. expires → datetime (formato MM/YYYY)
    df["expires"] = pd.to_datetime(df["expires"], format="%m/%Y", errors="coerce")

    # 3. acct_open_date → datetime
    df["acct_open_date"] = pd.to_datetime(
        df["acct_open_date"], format="%m/%Y", errors="coerce"
    )

    # 4. has_chip: YES/NO → bool
    df["has_chip"] = df["has_chip"].str.upper().map({"YES": True, "NO": False})

    # 5. card_on_dark_web: Yes/No → bool
    df["card_on_dark_web"] = df["card_on_dark_web"].str.strip().map(
        {"Yes": True, "No": False}
    )

    # 6. card_type y card_brand como categoría
    df["card_type"]  = df["card_type"].astype("category")
    df["card_brand"] = df["card_brand"].astype("category")

    # 7. Descartar filas sin id o client_id
    antes = len(df)
    df.dropna(subset=["id", "client_id"], inplace=True)
    descartadas = antes - len(df)

    # 8. Eliminar duplicados por id de tarjeta
    antes = len(df)
    df.drop_duplicates(subset=["id"], keep="first", inplace=True)
    descartadas += antes - len(df)

    print(f"\n   🗑️  Filas descartadas : {descartadas:,}")
    print(f"   ✓  Filas limpias     : {len(df):,}")
    reporte_basico(df, "cards (limpio)")
    guardar(df, "cards_clean")
    return df


# ══════════════════════════════════════════════
#  3. USERS_DATA.CSV  (~160 KB)
# ══════════════════════════════════════════════

def limpiar_usuarios():
    separador("3 · users_data.csv")
    df = pd.read_csv(USERS_CSV, low_memory=False)
    reporte_basico(df, "users (original)")

    # ── Limpieza ──────────────────────────────────────

    # 1. Columnas de dinero que vienen como "$29,278"
    money_cols = ["per_capita_income", "yearly_income", "total_debt"]
    for col in money_cols:
        if col in df.columns:
            # Conversión siempre forzada (compatible pandas 3.x + ArrowDtype)
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\$,\s]", "", regex=True)
                .replace({"nan": None, "": None})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. credit_score debe estar en rango razonable [300, 850]
    antes = len(df)
    df = df[
        df["credit_score"].between(300, 850, inclusive="both") |
        df["credit_score"].isna()
    ]
    print(f"   ✂️  Filas con credit_score inválido eliminadas: {antes - len(df)}")

    # 3. Edades razonables: current_age entre 18 y 110
    antes = len(df)
    df = df[df["current_age"].between(18, 110) | df["current_age"].isna()]
    print(f"   ✂️  Filas con edad inválida eliminadas: {antes - len(df)}")

    # 4. gender como categoría
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype("category")

    # 5. Latitud y longitud: coordenadas válidas
    if "latitude" in df.columns:
        df = df[
            df["latitude"].between(-90, 90) &
            df["longitude"].between(-180, 180)
        ]

    # 6. Eliminar duplicados por id de usuario
    antes = len(df)
    df.drop_duplicates(subset=["id"], keep="first", inplace=True)
    print(f"   ✂️  Duplicados eliminados: {antes - len(df)}")

    print(f"   ✓  Filas limpias : {len(df):,}")
    reporte_basico(df, "users (limpio)")
    guardar(df, "users_clean")
    return df


# ══════════════════════════════════════════════
#  4. TRAIN_FRAUD_LABELS.JSON  (~159 MB)
# ══════════════════════════════════════════════

def limpiar_labels():
    separador("4 · train_fraud_labels.json")
    print(f"   Tamaño original: {tamaño_mb(FRAUD_LABELS_JSON):.0f} MB")
    print("   Cargando JSON…")
    t0 = time.time()

    with open(FRAUD_LABELS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"   Tipo raíz del JSON : {type(raw).__name__}")
    print(f"   Registros cargados : {len(raw):,}  ({time.time()-t0:.1f}s)")

    # ── Detectar estructura del JSON ──
    # Soporta tres formatos:
    #   A) { "transaction_id": "Yes"/"No" }           → dict plano
    #   B) { "target": { "transaction_id": "Yes" } }  → dict anidado (este caso)
    #   C) [ {"transaction_id": 123, "is_fraud": ...} ] → lista de dicts
    if isinstance(raw, dict):
        # Verificar si tiene una clave envolvente como "target", "data", "labels"
        claves = list(raw.keys())
        print(f"   Claves raíz        : {claves[:5]}")
        primera_val = list(raw.values())[0]
        if isinstance(primera_val, dict):
            # Formato B: dict anidado → extraer el dict interno
            print(f"   Formato detectado  : dict anidado (clave wrapper: '{claves[0]}')")
            raw = primera_val
        else:
            print(f"   Formato detectado  : dict plano")
        df = pd.DataFrame(list(raw.items()), columns=["transaction_id", "is_fraud"])

    elif isinstance(raw, list):
        print(f"   Formato detectado  : lista de dicts")
        df = pd.DataFrame(raw)
        id_col    = next((c for c in df.columns if "id"    in c.lower()), df.columns[0])
        fraud_col = next((c for c in df.columns if "fraud" in c.lower()), df.columns[1])
        df = df.rename(columns={id_col: "transaction_id", fraud_col: "is_fraud"})
        df = df[["transaction_id", "is_fraud"]]
    else:
        raise ValueError(f"Estructura JSON inesperada: {type(raw)}")

    print(f"   Muestra parseada   : {list(zip(df['transaction_id'][:3], df['is_fraud'][:3]))}")

    # ── Convertir is_fraud a bool de forma robusta ──
    # Soporta: True/False, "Yes"/"No", "true"/"false", "1"/"0", 1/0
    def to_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in ("yes", "true", "1")
        return None  # nulo si no reconocible

    df["is_fraud"] = df["is_fraud"].apply(to_bool)

    # ── transaction_id a int ──
    df["transaction_id"] = pd.to_numeric(df["transaction_id"], errors="coerce")

    # ── Eliminar nulos y duplicados ──
    antes = len(df)
    df.dropna(subset=["transaction_id", "is_fraud"], inplace=True)
    df["transaction_id"] = df["transaction_id"].astype("int64")
    df.drop_duplicates(subset=["transaction_id"], keep="first", inplace=True)
    descartados = antes - len(df)
    print(f"   🗑️  Registros descartados : {descartados:,}")
    print(f"   ✓  Registros limpios     : {len(df):,}")

    # ── Distribución de clases ──
    n_total  = len(df)
    if n_total == 0:
        print("   ⚠️  ADVERTENCIA: el DataFrame quedó vacío tras la limpieza.")
        print("      Revisá que el archivo JSON tenga el formato correcto.")
        return df

    n_fraude = int(df["is_fraud"].sum())
    print(f"\n   📈 Distribución de clases:")
    print(f"      Fraude     : {n_fraude:,}  ({n_fraude/n_total*100:.2f}%)")
    print(f"      No fraude  : {n_total-n_fraude:,}  ({(n_total-n_fraude)/n_total*100:.2f}%)")
    if n_fraude / n_total < 0.1:
        print(f"      ⚠️  Desbalance severo — considerar SMOTE o class_weight en el modelo")

    reporte_basico(df, "fraud_labels (limpio)")
    out = guardar(df, "fraud_labels_clean")

    orig_mb  = tamaño_mb(FRAUD_LABELS_JSON)
    clean_mb = tamaño_mb(out)
    print(f"   📉 Compresión: {orig_mb:.0f} MB → {clean_mb:.0f} MB  "
          f"({(1 - clean_mb/orig_mb)*100:.0f}% reducción)")
    return df


# ══════════════════════════════════════════════
#  5. MCC_CODES.JSON  (solo ~5 KB, sin limpieza)
# ══════════════════════════════════════════════

def procesar_mcc():
    separador("5 · mcc_codes.json")
    with open(MCC_CODES_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(list(raw.items()), columns=["code", "description"])

    # Agregar agrupación de alto nivel según rangos estándar MCC
    def grupo_mcc(code: str) -> str:
        try:
            c = int(code)
        except ValueError:
            return "Otro"
        if 1000 <= c <= 1799: return "Construcción"
        if 2000 <= c <= 3999: return "Manufactura / Transporte"
        if 4000 <= c <= 4999: return "Transporte / Servicios Públicos"
        if 5000 <= c <= 5999: return "Retail / Comercio"
        if 6000 <= c <= 6999: return "Finanzas / Seguros"
        if 7000 <= c <= 7999: return "Hotelería / Entretenimiento"
        if 8000 <= c <= 8999: return "Servicios Profesionales / Salud"
        if 9000 <= c <= 9999: return "Gobierno / Servicios Públicos"
        return "Otro"

    df["category_group"] = df["code"].apply(grupo_mcc)
    print(f"   Códigos MCC cargados: {len(df)}")
    print(f"   Grupos:")
    for g, n in df["category_group"].value_counts().items():
        print(f"     • {g}: {n}")

    out = CLEAN_DIR / "mcc_codes_clean.json"
    df.to_json(out, orient="records", indent=2, force_ascii=False)
    print(f"   💾 Guardado → {out}")
    return df


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def main():
    print("\n" + "▓" * 60)
    print("  PREPARACIÓN DE DATOS · Proyecto Fraude · 2026")
    print("▓" * 60)

    resultados = {}

    # ── Ejecutar cada pipeline ──
    if TRANSACTIONS_CSV.exists():
        resultados["transactions"] = limpiar_transacciones()
    else:
        print(f"\n⚠️  No se encontró {TRANSACTIONS_CSV}")
        print("   Descargá el archivo desde Google Drive y colocalo en ./data/raw/")

    if CARDS_CSV.exists():
        resultados["cards"] = limpiar_tarjetas()
    else:
        print(f"\n⚠️  No se encontró {CARDS_CSV}")

    if USERS_CSV.exists():
        resultados["users"] = limpiar_usuarios()
    else:
        print(f"\n⚠️  No se encontró {USERS_CSV}")

    if FRAUD_LABELS_JSON.exists():
        resultados["labels"] = limpiar_labels()
    else:
        print(f"\n⚠️  No se encontró {FRAUD_LABELS_JSON}")

    if MCC_CODES_JSON.exists():
        resultados["mcc"] = procesar_mcc()
    else:
        print(f"\n⚠️  No se encontró {MCC_CODES_JSON}")

    # ── Resumen final ──
    separador("RESUMEN FINAL")
    archivos_clean = list(CLEAN_DIR.iterdir())
    total_mb = sum(f.stat().st_size for f in archivos_clean) / 1_048_576
    print(f"\n   Archivos generados en {CLEAN_DIR}:")
    for f in sorted(archivos_clean):
        print(f"     • {f.name:<40} {f.stat().st_size/1_048_576:>7.1f} MB")
    print(f"\n   Total ocupado en disco: {total_mb:.1f} MB")
    print("\n   ✅ Preparación completada.\n")


if __name__ == "__main__":
    main()
