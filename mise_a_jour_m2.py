# ────────────────────────────────────────────────────────────
# fichier : appairage_m2_streamlit.py
# ────────────────────────────────────────────────────────────
"""Outil Streamlit d'appairage entre codes M2 et familles client — version 2025‑07‑03.

Nouveauté : **plus aucun bug de relecture** après un rerun !
-----------------------------------------------------------
Le problème provenait du fait que l’objet `UploadedFile` peut perdre sa
référence au flux sous‑jacent après un rafraîchissement Streamlit. On fige donc
*immédiatement* son contenu binaire (`bytes`) dans `st.session_state` — on ne
stocke plus l’objet brut.

Fonctionnement
==============
1. L’utilisateur dépose un ou plusieurs fichiers par lot.
2. Chaque fichier est converti en dict `{"name": <str>, "bytes": <bytes>}` et
   conservé en session.
3. À chaque rerun, la lecture se fait **à partir des bytes** — jamais plus sur
   l’objet Streamlit.

Résultats produits :
• `appairage_M2_CodeFamilleClient_YYMMDD.csv` : M2_nouveau → code famille client
  (majoritaire).
• `M2_MisAJour_YYMMDD.csv` : M2_nouveau → M2_ancien (majoritaire).
"""
from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List

import pandas as pd
import streamlit as st

# ────────────────────────────  Types  ────────────────────────────
class StoredFile(TypedDict):
    name: str
    bytes: bytes


# ────────────────────────────  Page  ────────────────────────────
st.set_page_config("Appairage M2", "🛠️", layout="wide")
st.title("🛠️  Appairage codes M2 / familles client")

# ───────────────────────  Fonctions utilitaires  ───────────────────────

def read_csv_buffer(buf: io.BytesIO) -> pd.DataFrame:
    """Essaye utf‑8, latin‑1, cp1252 + détection de séparateur."""
    for enc in ("utf-8", "latin1", "cp1252"):
        buf.seek(0)
        try:
            sample = buf.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
        except (UnicodeDecodeError, csv.Error, pd.errors.ParserError):
            continue
    raise ValueError("CSV illisible : encodage/separateur non reconnu.")


def read_any(file: StoredFile) -> pd.DataFrame | None:
    """Retourne un DataFrame depuis un fichier stocké (dict name/bytes)."""
    name = file["name"].lower()
    data = file["bytes"]
    buf = io.BytesIO(data)
    suffix = Path(name).suffix

    # CSV ----------------------------------------------------------------
    if suffix == ".csv":
        try:
            return read_csv_buffer(buf)
        except ValueError as err:
            st.error(f"{name} : {err}")
            return None

    # EXCEL --------------------------------------------------------------
    if suffix in {".xlsx", ".xls"}:
        buf.seek(0)
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        try:
            return pd.read_excel(buf, engine=engine)
        except ImportError:
            st.error("Le format .xls nécessite `xlrd<2.0.0`. Installe‑le puis relance.")
            return None
        except Exception as exc:
            st.error(f"Erreur Excel ({name}) : {exc}")
            return None

    # PARQUET ------------------------------------------------------------
    if suffix == ".parquet":
        buf.seek(0)
        try:
            return pd.read_parquet(buf)
        except Exception as exc:
            st.error(f"Erreur Parquet ({name}) : {exc}")
            return None

    st.error(f"Extension non prise en charge : {suffix}")
    return None


def to_m2_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)


def add_cols(df: pd.DataFrame, ref_idx: int, m2_idx: int, ref_label: str, m2_label: str) -> pd.DataFrame:
    out = df.copy()
    out[ref_label] = out.iloc[:, ref_idx - 1].astype(str)
    out[m2_label] = to_m2_series(out.iloc[:, m2_idx - 1])
    return out


def idx_ok(df: pd.DataFrame, idx: int) -> bool:
    return 1 <= idx <= df.shape[1]


# ──────────────────────  Configuration des lots  ──────────────────────
LOTS = {
    "old": ("Données N‑1", "Idx Réf. client", "Idx Code M2 ancien"),
    "new": ("Données N", "Idx Réf. client", "Idx Code M2 nouveau"),
    "map": ("Table d'appairage", "Idx Code M2 ancien", "Idx Code famille client"),
}

# -----------------------  Session State  ------------------------------
for key in LOTS:
    st.session_state.setdefault(f"{key}_files", [])   # type: List[StoredFile]
    st.session_state.setdefault(f"{key}_names", [])

# ----------------------  Interface Upload  ----------------------------
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
                    st.session_state[f"{key}_files"].append({"name": up.name, "bytes": up.getvalue()})
                    st.session_state[f"{key}_names"].append(up.name)
                    new += 1
            if new:
                st.success(f"{new} fichier(s) ajouté(s)")

        st.number_input(lab_ref, 1, 50, 1, key=f"{key}_ref")
        st.number_input(lab_val, 1, 50, 2, key=f"{key}_val")
        st.caption(f"{len(st.session_state[f'{key}_files'])} fichier(s) chargés")

# ─────────────────────────────  Traitement  ─────────────────────────────
if st.button("🔗  Lancer l'appairage"):
    # ---------- Vérifications ----------
    if not all(st.session_state[f"{k}_files"] for k in LOTS):
        st.warning("Merci de charger les trois lots de données avant de continuer.")
        st.stop()

    # ---------- Lecture fichiers ----------
    dfs: dict[str, List[pd.DataFrame]] = {}
    for key in LOTS:
        dfs[key] = [read_any(f) for f in st.session_state[f"{key}_files"]]
        if any(df is None for df in dfs[key]):
            st.error("Erreur de lecture dans au moins un fichier — corrige puis réessaie.")
            st.stop()

    old_raw = pd.concat(dfs["old"], ignore_index=True).drop_duplicates()
    new_raw = pd.concat(dfs["new"], ignore_index=True).drop_duplicates()
    map_raw = pd.concat(dfs["map"], ignore_index=True).drop_duplicates()

    # ---------- Vérif index ----------
    for df, key in ((old_raw, "old"), (new_raw, "new"), (map_raw, "map")):
        if not idx_ok(df, st.session_state[f"{key}_ref"]) or not idx_ok(df, st.session_state[f"{key}_val"]):
            st.error(f"Index hors limites pour le lot {key.upper()}.")
            st.stop()

    # ---------- Pré‑traitement ----------
    old_df = add_cols(old_raw, st.session_state["old_ref"], st.session_state["old_val"], "Reference", "M2_ancien")
    new_df = add_cols(new_raw, st.session_state["new_ref"], st.session_state["new_val"], "Reference", "M2_nouveau")

    map_df = map_raw.copy()
    map_df["M2_ancien"] = to_m2_series(map_df.iloc[:, st.session_state["map_ref"] - 1])
    map_df["Code_famille_Client"] = map_df.iloc[:, st.session_state["map_val"] - 1].astype(str)
    map_df = map_df[["M2_ancien", "Code_famille_Client"]]

    # ---------- Fusions ----------
    with st.spinner("Fusion en cours…"):
        merged = new_df.merge(old_df[["Reference", "M2_ancien"]], on="Reference", how="outer")
        merged = merged.merge(map_df, on="M2_ancien", how="outer")

    st.success("✅ Fusion terminée")
    st.dataframe(merged.head())

    # ---------- Appairages ----------
    family_map = (
        merged.groupby("M2_nouveau")["Code_famille_Client"]
        .agg(lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA)
        .reset_index()
    )

    m2_map = (
        merged.groupby("M2_nouveau")["M2_ancien"]
        .agg(lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA)
        .reset_index()
    )

    appairage_df = family_map.rename(columns={"Code_famille_Client": "Code_famille_Client"})
    m2_update_df = m2_map.rename(columns={"M2_ancien": "M2_ancien"})[["M2_ancien", "M2_nouveau"]]

    # ---------- Téléchargements ----------
    dstr = datetime.today().strftime("%y%m%d")
    st.download_button("⬇️ Appairage M2 → Famille", appairage_df.to_csv(index=False, sep=";"), file_name=f"appairage_M2_CodeFamilleClient_{dstr}.csv", mime="text/csv")
    st.download_button("⬇️ Mise à jour M2", m2_update_df.to_csv(index=False, sep=";"), file_name=f"M2_MisAJour_{dstr}.csv", mime="text/csv")

    st.markdown(
        f"**{len(merged):,}** lignes fusionnées — "
        f"codes famille déterminés pour **{appairage_df['Code_famille_Client'].notna().sum():,}** M2_nouveau"
    )
