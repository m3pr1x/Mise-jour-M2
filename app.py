# ──────────────────────────────────────────────────────────
# streamlit_app.py  – Pages « Mise à jour M2 » (PC & Appairage)
# ──────────────────────────────────────────────────────────
"""Deux workflows :
1. **PC : Mise à jour des codes M2** (2 fichiers N‑1 + N) → `M2_MisAJour_YYMMDD.csv`
2. **Client : Appairage M2 ⇆ famille** (N‑1 + N + mapping) → `appairage_M2_CodeFamilleClient_YYMMDD.csv`
Les uploads persistent en session jusqu’au « Rerun & clear » manuel.

`psutil` est optionnel : ajoute‑le dans *requirements.txt* si tu veux la jauge
mémoire, sinon elle s’affiche « n/a ».
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
    psutil = None  # on masque juste la RAM

DEBUG_SAMPLE: int | None = None  # mettre 10_000 pour tester vite

st.set_page_config("Appairage / Mise à jour M2", "🛠️", layout="wide")
page = st.sidebar.radio("Navigation", ["Mise à jour M2 – PC", "Mise à jour M2 – Appairage client"], key="nav")

# ═════════════  FONCTIONS UTILITAIRES  ═════════════

def _read_csv(buf: io.BytesIO) -> pd.DataFrame:
    for enc in ("utf-8", "latin1", "cp1252"):
        buf.seek(0)
        try:
            sample = buf.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, encoding=enc, dtype=str, low_memory=True, on_bad_lines="skip")
        except (UnicodeDecodeError, csv.Error, pd.errors.ParserError):
            continue
    raise ValueError("CSV illisible : encodage/séparateur introuvable")

@st.cache_data(show_spinner=False, hash_funcs={io.BytesIO: lambda _: None})
def load_df(up) -> pd.DataFrame:
    ext = Path(up.name.lower()).suffix
    buf = io.BytesIO(up.getvalue())
    if ext == ".csv":
        return _read_csv(buf)
    if ext == ".xlsx":
        buf.seek(0)
        return pd.read_excel(buf, engine="openpyxl", dtype=str)
    raise ValueError(f"Extension {ext} non supportée")


def to_m2(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)


def add_cols(df: pd.DataFrame, ref_i: int, m2_i: int, ref_label: str, m2_label: str) -> pd.DataFrame:
    out = df.iloc[:, [ref_i-1, m2_i-1]].copy()
    out.columns = [ref_label, m2_label]
    out[m2_label] = to_m2(out[m2_label])
    return out


def ram() -> str:
    if psutil is None:
        return "n/a"
    return f"{psutil.Process(os.getpid()).memory_info().rss/1_048_576:,.0f} Mo"

# ═════════════  COMPOSANT UPLOADER  ═════════════

def uploader(prefix: str, lots: Dict[str, tuple[str, str, str]]):
    for k in lots:
        st.session_state.setdefault(f"{prefix}_{k}_files", [])
        st.session_state.setdefault(f"{prefix}_{k}_names", [])

    cols = st.columns(len(lots))
    for (k, (title, lab_ref, lab_val)), col in zip(lots.items(), cols):
        with col:
            st.subheader(title)
            ups = st.file_uploader("Déposer…", type=("csv", "xlsx"), accept_multiple_files=True, key=f"uploader_{prefix}_{k}")
            if ups:
                new = 0
                for up in ups:
                    if up.name not in st.session_state[f"{prefix}_{k}_names"]:
                        st.session_state[f"{prefix}_{k}_files"].append(up)
                        st.session_state[f"{prefix}_{k}_names"].append(up.name)
                        new += 1
                if new:
                    st.success(f"{new} fichier(s) ajouté(s)")

            st.number_input(lab_ref, 1, 50, 1, key=f"{prefix}_{k}_ref")
            st.number_input(lab_val, 1, 50, 2, key=f"{prefix}_{k}_val")
            st.caption(f"{len(st.session_state[f'{prefix}_{k}_files'])} fichier(s) | RAM : {ram()}")

# ═════════════  PAGE PC  ═════════════
if page.startswith("Mise à jour M2 – PC"):
    st.header("🔄 Mise à jour des codes M2 (Personal Catalogue)")

    LOTS_PC = {
        "old": ("Données N‑1", "Réf. produit", "M2 ancien"),
        "new": ("Données N",   "Réf. produit", "M2 nouveau"),
    }
    uploader("pc", LOTS_PC)

    if st.button("🚀 Générer le fichier M2_MisAJour", key="run_pc"):
        if not all(st.session_state[f"pc_{k}_files"] for k in LOTS_PC):
            st.warning("Merci de charger N‑1 et N.")
            st.stop()

        dfs = {}
        for k in LOTS_PC:
            parts = [load_df(f) for f in st.session_state[f"pc_{k}_files"]]
            if any(p is None for p in parts):
                st.error("Erreur de lecture.")
                st.stop()
            df = pd.concat(parts, ignore_index=True).drop_duplicates()
            if DEBUG_SAMPLE:
                df = df.head(DEBUG_SAMPLE)
            dfs[k] = df

        for k, df in dfs.items():
            r_i, m_i = st.session_state[f"pc_{k}_ref"], st.session_state[f"pc_{k}_val"]
            if not (1 <= r_i <= df.shape[1] and 1 <= m_i <= df.shape[1]):
                st.error(f"Indices hors plage ({k}).")
                st.stop()

        old_df = add_cols(dfs["old"], st.session_state["pc_old_ref"], st.session_state["pc_old_val"], "Ref", "M2_ancien")
        new_df = add_cols(dfs["new"], st.session_state["pc_new_ref"], st.session_state["pc_new_val"], "Ref", "M2_nouveau")

        merged = new_df.merge(old_df[["Ref", "M2_ancien"]], on="Ref", how="left")
        maj = merged.groupby("M2_nouveau")["M2_ancien"].agg(lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA).reset_index()

        dstr = datetime.today().strftime("%y%m%d")
        st.download_button("⬇️ Télécharger M2_MisAJour.csv", maj.to_csv(index=False, sep=";"), file_name=f"M2_MisAJour_{dstr}.csv", mime="text/csv")
        st.success("Fichier généré.")
        st.dataframe(maj.head())

# ═════════════  PAGE Appairage  ═════════════
if page.startswith("Mise à jour M2 – Appairage"):
    st.header("🔗 Appairage M2 / famille client")

    LOTS_CL = {
        "old": ("Données N‑1",   "Réf. produit", "M2 ancien"),
        "new": ("Données N",     "Réf. produit", "M2 nouveau"),
        "map": ("Table mapping", "M2 ancien",   "Code famille client"),
    }
    uploader("cl", LOTS_CL)

    if st.button("🚀 Générer l'appairage", key="run_cl"):
        if not all(st.session_state[f"cl_{k}_files"] for k in LOTS_CL):
            st.warning("Merci de charger les 3 fichiers.")
            st.stop()

        dfs = {}
        for k in LOTS_CL:
            parts = [load_df(f) for f in st.session_state[f"cl_{k}_files"]]
            if any(p is None
