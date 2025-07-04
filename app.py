# ───────────────────────────────────────────────────────────
# streamlit_app.py  – « no‑psutil », deux modules séparés
# ───────────────────────────────────────────────────────────
"""
1️⃣ **Mise à jour M2 – Personal Catalogue**
    • charge *N‑1* + *N*   →  génère **M2_MisAJour.csv**
2️⃣ **Mise à jour M2 – Appairage Client**
    • charge *N‑1* + *N* + *Mapping* → génère **appairage_M2_CodeFamilleClient.csv**

Les fichiers uploadés restent en session jusqu’au « Reset » (menu R 🔄).
`psutil` est optionnel : ajoute‑le dans *requirements.txt* si tu veux la jauge RAM.
"""
from __future__ import annotations
import csv, io, os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

DEBUG_SAMPLE: int | None = None   # 10_000 pour debug

st.set_page_config("Appairage / Mise à jour M2", "🛠️", layout="wide")
page = st.sidebar.radio("Navigation", ["Mise à jour M2 – PC", "Mise à jour M2 – Appairage client"])

# ──────────────────────────  Fonctions utilitaires  ──────────────────────────

def _read_csv(buf: io.BytesIO) -> pd.DataFrame:
    for enc in ("utf-8", "latin1", "cp1252"):
        buf.seek(0)
        try:
            sample = buf.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, encoding=enc, dtype=str, low_memory=True, on_bad_lines="skip")
        except (csv.Error, UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("CSV illisible : encodage/séparateur non détecté")

@st.cache_data(show_spinner=False, hash_funcs={io.BytesIO: lambda _: None})
def load_df(file) -> pd.DataFrame:
    ext = Path(file.name.lower()).suffix
    buf = io.BytesIO(file.getvalue())
    if ext == ".csv":
        return _read_csv(buf)
    if ext == ".xlsx":
        buf.seek(0)
        return pd.read_excel(buf, engine="openpyxl", dtype=str)
    raise ValueError(f"Extension {ext} non gérée")


def to_m2(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)


def add_cols(df: pd.DataFrame, ref_i: int, m2_i: int, ref_name: str, m2_name: str) -> pd.DataFrame:
    out = df.iloc[:, [ref_i-1, m2_i-1]].copy()
    out.columns = [ref_name, m2_name]
    out[m2_name] = to_m2(out[m2_name])
    return out


def mem() -> str:
    if psutil is None:
        return "n/a"
    return f"{psutil.Process(os.getpid()).memory_info().rss/1_048_576:,.0f} Mo"

# ──────────────────────────  Interface uploads  ────────────────────────────

def uploader(section_key: str, lots: Dict[str, tuple[str, str, str]]):
    """Construit l’UI d’upload pour la page courante."""
    for k in lots:
        st.session_state.setdefault(f"{section_key}_{k}_files", [])
        st.session_state.setdefault(f"{section_key}_{k}_names", [])

    cols = st.columns(len(lots))
    for (key, (title, lab_ref, lab_val)), col in zip(lots.items(), cols):
        with col:
            st.subheader(title)
            uploads = st.file_uploader("Ajouter fichier(s)…", type=("csv", "xlsx"),
                                       accept_multiple_files=True, key=f"uploader_{section_key}_{key}")
            if uploads:
                fresh = 0
                for up in uploads:
                    if up.name not in st.session_state[f"{section_key}_{key}_names"]:
                        st.session_state[f"{section_key}_{key}_files"].append(up)
                        st.session_state[f"{section_key}_{key}_names"].append(up.name)
                        fresh += 1
                if fresh:
                    st.success(f"{fresh} fichier(s) ajouté(s)")

            st.number_input(lab_ref, 1, 50, 1, key=f"{section_key}_{key}_ref")
            st.number_input(lab_val, 1, 50, 2, key=f"{section_key}_{key}_val")
            st.caption(f"{len(st.session_state[f'{section_key}_{key}_files'])} fichier(s) | RAM : {mem()}")

# ──────────────────────────  Page 1 : Mise à jour PC  ───────────────────────
if page.startswith("Mise à jour M2 – PC"):
    st.header("🔄 Mise à jour codes M2 – Personal Catalogue")

    LOTS_PC = {
        "old": ("Données N‑1", "Réf. client", "M2 ancien"),
        "new": ("Données N",   "Réf. client", "M2 nouveau"),
    }
    uploader("pc", LOTS_PC)

    if st.button("🚀 Générer M2_MisAJour", key="run_pc"):
        if not all(st.session_state[f"pc_{k}_files"] for k in LOTS_PC):
            st.warning("Chargez les deux lots.")
            st.stop()

        # charge & concat -------------------------------------------------
        dfs = {}
        for k in LOTS_PC:
            parts = [load_df(f) for f in st.session_state[f"pc_{k}_files"]]
            if any(p is None for p in parts):
                st.error("Erreur de lecture dans un fichier.")
                st.stop()
            df = pd.concat(parts, ignore_index=True).drop_duplicates()
            if DEBUG_SAMPLE:
                df = df.head(DEBUG_SAMPLE)
            dfs[k] = df

        # index ok ? ------------------------------------------------------
        for k, df in dfs.items():
            ref_i, m2_i = st.session_state[f"pc_{k}_ref"], st.session_state[f"pc_{k}_val"]
            if not (1 <= ref_i <= df.shape[1] and 1 <= m2_i <= df.shape[1]):
                st.error(f"Index hors plage pour {k.upper()}")
                st.stop()

        old_df = add_cols(dfs["old"], st.session_state["pc_old_ref"], st.session_state["pc_old_val"], "Ref", "M2_ancien")
        new_df = add_cols(dfs["new"], st.session_state["pc_new_ref"], st.session_state["pc_new_val"], "Ref", "M2_nouveau")

        # majorité M2 ancien par M2 nouveau ------------------------------
        merged = new_df.merge(old_df[["Ref", "M2_ancien"]], on="Ref", how="left")
        rel = merged.groupby("M2_nouveau")["M2_ancien"].agg(lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA).reset_index()

        dstr = datetime.today().strftime("%y%m%d")
        st.download_button("⬇️ M2_MisAJour.csv", rel.to_csv(index=False, sep=";"),
                           file_name=f"M2_MisAJour_{dstr}.csv", mime="text/csv")
        st.success("Fichier généré ! (ligne d’exemple ci‑dessous)")
        st.dataframe(rel.head())

# ──────────────────────────  Page 2 : Appairage client  ─────────────────────
if page.startswith("Mise à jour M2 – Appairage"):
    st.header("🔗 Appairage codes M2 / familles client")

    LOTS_CL = {
        "old": ("Données N‑1",   "Réf. client", "M2 ancien"),
        "new": ("Données N",     "Réf. client", "M2 nouveau"),
        "map": ("Table mapping", "M2 ancien",   "Code famille client"),
    }
    uploader("cl", LOTS_CL)

    if st.button("🚀 Générer appairage", key="run_cl"):
        if not all(st.session_state[f"cl_{k}_files"] for k in LOTS_CL):
            st.warning("Chargez les trois lots.")
            st.stop()

        dfs = {}
        for k in LOTS_CL:
            parts = [load_df(f) for f in st.session_state[f"cl_{k}_files"]]
            if any(p is None for p in parts):
                st.error("Erreur de lecture.")
                st.stop()
            df = pd.concat(parts, ignore_index=True).drop_duplicates()
            if DEBUG_SAMPLE:
                df = df.head(DEBUG_SAMPLE)
            dfs[k] = df

        # index check
        for k, df in dfs.items():
            ref_i, val_i = st.session_state[f"cl_{k}_ref"], st.session_state[f"cl_{k}_val"]
            if not (1 <= ref_i <= df.shape[1] and 1 <= val_i <= df.shape[ )
