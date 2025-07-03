# ────────────────────────────────────────────────────────────
# fichier : appairage_m2_streamlit.py
# ────────────────────────────────────────────────────────────
"""Outil Streamlit d'appairage entre codes M2 et familles client.

Trois lots de fichiers peuvent être glissés‑déposés :
  1. Données N‑1     → colonnes : Référence client  |  Code M2 ancien
  2. Données N       → colonnes : Référence client  |  Code M2 nouveau
  3. Table d'appairage → colonnes : Code M2 ancien  |  Code famille client

L'utilisateur renseigne la position (1‑indexée) des colonnes utiles pour chaque
lot. Plusieurs fichiers peuvent être concaténés pour un même lot.

Résultats produits :
  • Fichier « appairage_M2_CodeFamilleClient_YYMMDD.csv » :
      pour chaque M2_nouveau, le code famille client majoritaire (si connu).
  • Fichier « M2_MisAJour_YYMMDD.csv » :
      pour chaque M2_nouveau, le M2_ancien majoritaire (si connu).
  • Visualisation du DataFrame fusionné (outer → Référence, puis outer → M2_ancien).

➡️  Correctif intégré : le bug « fichier vide au 2ᵉ run » est résolu en figeant
    immédiatement le contenu binaire de chaque `UploadedFile` (via
    `upload.getvalue()`) avant toute lecture.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ────────────────────────────  Page  ────────────────────────────
st.set_page_config("Appairage M2", "🛠️", layout="wide")
st.title("🛠️  Appairage codes M2 / familles client")

# ───────────────────────  Fonctions utilitaires  ───────────────────────


def read_csv_buffer(buf: io.BytesIO) -> pd.DataFrame:
    """Lecture robuste d'un CSV : détecte séparateur + essaie plusieurs encodages."""
    for enc in ("utf-8", "latin1", "cp1252"):
        buf.seek(0)
        try:
            sample = buf.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
        except (UnicodeDecodeError, csv.Error, pd.errors.ParserError):
            continue
    raise ValueError("Fichier CSV illisible : encodage ou séparateur non reconnu.")


def read_any(upload) -> pd.DataFrame | None:
    """Retourne un DataFrame depuis un UploadFile (CSV / Excel / Parquet).

    ❗️ Correctif bug Streamlit :
        - On fige le contenu binaire dès l'appel (`data = upload.getvalue()`)
        - On travaille ensuite sur **notre** `BytesIO` réinitialisable → plus
          de problème de curseur lors des reruns.
    """
    if upload is None:
        return None

    data: bytes = upload.getvalue()  # fige le contenu dans un buffer immuable
    buf = io.BytesIO(data)           # crée un flux repositionnable
    suffix = Path(upload.name.lower()).suffix

    # CSV ----------------------------------------------------------------
    if suffix == ".csv":
        try:
            return read_csv_buffer(buf)
        except ValueError as err:
            st.error(str(err))
            return None

    # EXCEL --------------------------------------------------------------
    if suffix in {".xlsx", ".xls"}:
        buf.seek(0)
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        try:
            return pd.read_excel(buf, engine=engine)
        except ImportError:
            st.error(
                "Le format .xls nécessite le paquet `xlrd<2.0.0`. "
                "Installe‑le puis relance l'application."
            )
            return None
        except Exception as exc:  # garde‑fou supplémentaire
            st.error(f"Erreur de lecture Excel : {exc}")
            return None

    # PARQUET ------------------------------------------------------------
    if suffix == ".parquet":
        buf.seek(0)
        try:
            return pd.read_parquet(buf)
        except Exception as exc:
            st.error(f"Erreur de lecture Parquet : {exc}")
            return None

    st.error(f"Extension non prise en charge : {suffix}")
    return None


def to_m2_series(s: pd.Series) -> pd.Series:
    """Normalise les codes M2 → chaîne sur 6 caractères (zéro‑padding)."""
    return s.astype(str).str.zfill(6)


def add_cols(df: pd.DataFrame, ref_idx: int, m2_idx: int, ref_label: str, m2_label: str) -> pd.DataFrame:
    """Ajoute deux colonnes normalisées pour la référence et le code M2."""
    out = df.copy()
    out[ref_label] = out.iloc[:, ref_idx - 1].astype(str)
    out[m2_label] = to_m2_series(out.iloc[:, m2_idx - 1])
    return out


def idx_ok(df: pd.DataFrame, idx: int) -> bool:
    """Vérifie que l'index renseigné existe bien dans le DataFrame."""
    return 1 <= idx <= df.shape[1]


# ──────────────────────  Configuration des lots  ──────────────────────
LOTS = {
    "old": (
        "Données N‑1",
        "Idx Réf. client",
        "Idx Code M2 ancien",
    ),
    "new": (
        "Données N",
        "Idx Réf. client",
        "Idx Code M2 nouveau",
    ),
    "map": (
        "Table d'appairage",
        "Idx Code M2 ancien",
        "Idx Code famille client",
    ),
}

# Initialisation de la session -----------------------------------------
for key in LOTS:
    st.session_state.setdefault(f"{key}_files", [])   # UploadedFile stockés
    st.session_state.setdefault(f"{key}_names", [])   # noms pour éviter doublons

# Interface de chargement ----------------------------------------------
cols = st.columns(3)
for (key, (title, lab_ref, lab_val)), col in zip(LOTS.items(), cols):
    with col:
        st.subheader(title)
        uploads = st.file_uploader(
            "Glisser / déposer ou parcourir…",
            accept_multiple_files=True,
            type=("csv", "xlsx", "xls", "parquet"),
            key=f"uploader_{key}",
        )
        if uploads:
            new = 0
            for up in uploads:
                if up.name not in st.session_state[f"{key}_names"]:
                    st.session_state[f"{key}_files"].append(up)
                    st.session_state[f"{key}_names"].append(up.name)
                    new += 1
            if new:
                st.success(f"{new} fichier(s) ajouté(s)")

        st.number_input(lab_ref, 1, 50, 1, key=f"{key}_ref")
        st.number_input(lab_val, 1, 50, 2, key=f"{key}_val")
        st.caption(f"{len(st.session_state[f'{key}_files'])} fichier(s) chargés")

# ─────────────────────────────  Traitement  ─────────────────────────────
if st.button("🔗  Lancer l'appairage"):
    # Vérifications préliminaires --------------------------------------
    if not all(st.session_state[f"{k}_files"] for k in LOTS):
        st.warning("Merci de charger les trois lots de données avant de continuer.")
        st.stop()

    # Lecture des fichiers → DataFrames --------------------------------
    dfs: dict[str, list[pd.DataFrame]] = {}
    for key in LOTS:
        dfs[key] = [read_any(up) for up in st.session_state[f"{key}_files"]]
        if any(df is None for df in dfs[key]):
            st.error("Erreur de lecture dans au moins un fichier ; corrige puis réessaie.")
            st.stop()

    old_raw = pd.concat(dfs["old"], ignore_index=True).drop_duplicates()
    new_raw = pd.concat(dfs["new"], ignore_index=True).drop_duplicates()
    map_raw = pd.concat(dfs["map"], ignore_index=True).drop_duplicates()

    # Vérification des index -------------------------------------------
    for df, key in ((old_raw, "old"), (new_raw, "new"), (map_raw, "map")):
        if not idx_ok(df, st.session_state[f"{key}_ref"]) or not idx_ok(df, st.session_state[f"{key}_val"]):
            st.error(f"Index hors limites pour le lot {key.upper()}.")
            st.stop()

    # Pré‑traitement : normalisation des colonnes ----------------------
    old_df = add_cols(
        old_raw,
        st.session_state["old_ref"],
        st.session_state["old_val"],
        "Reference",
        "M2_ancien",
    )

    new_df = add_cols(
        new_raw,
        st.session_state["new_ref"],
        st.session_state["new_val"],
        "Reference",
        "M2_nouveau",
    )

    map_df = map_raw.copy()
    map_df["M2_ancien"] = to_m2_series(map_df.iloc[:, st.session_state["map_ref"] - 1])
    map_df["Code_famille_Client"] = map_df.iloc[:, st.session_state["map_val"] - 1].astype(str)
    map_df = map_df[["M2_ancien", "Code_famille_Client"]]

    # Fusions successives ---------------------------------------------
    with st.spinner("Fusion des fichiers…"):
        merged = new_df.merge(
            old_df[["Reference", "M2_ancien"]],
            on="Reference",
            how="outer",
        )
        merged = merged.merge(
            map_df,
            on="M2_ancien",
            how="outer",
        )

    st.success("✅  Fusion terminée")
    st.dataframe(merged.head())

    # ──────────  Dataset 1 : M2_nouveau → Code famille client  ──────────
    family_map = (
        merged.groupby("M2_nouveau")["Code_famille_Client"]
        .agg(lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA)
        .reset_index()
        .rename(columns={"Code_famille_Client": "Code_famille_Client"})
    )

    # ──────────  Dataset 2 : M2_nouveau → M2_ancien majoritaire  ─────────
    m2_map = (
        merged.groupby("M2_nouveau")["M2_ancien"]
        .agg(lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA)
        .reset_index()
        .rename(columns={"M2_ancien": "M2_ancien"})
    )

    # Ajustement des colonnes -----------------------------------------
    appairage_df = family_map[["M2_nouveau", "Code_famille_Client"]]
    m2_update_df = m2_map[["M2
