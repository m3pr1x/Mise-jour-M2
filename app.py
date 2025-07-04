# streamlit_app.py — Mise à jour M2 (PC) & Appairage client
# v2025‑07‑05 : boutons de téléchargement persistants + liste « à remplir »
from __future__ import annotations
import csv, io, os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import streamlit as st
try:
    import psutil
except ModuleNotFoundError:
    psutil = None

DEBUG_SAMPLE: int | None = None  # ex : 10_000

st.set_page_config(page_title="Mise a jour M2", page_icon="🛠", layout="wide")
page = st.sidebar.radio("Navigation", ("Mise a jour M2 - PC", "Mise a jour M2 - Appairage client"))

# ──────────── helpers de lecture ────────────

def read_csv(buf: io.BytesIO) -> pd.DataFrame:
    for enc in ("utf-8", "latin1", "cp1252"):
        buf.seek(0)
        try:
            sample = buf.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, encoding=enc, dtype=str, low_memory=True, on_bad_lines="skip")
        except (csv.Error, UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("CSV illisible")

@st.cache_data(show_spinner=False, hash_funcs={io.BytesIO: lambda _: None})
def load_df(up) -> pd.DataFrame:
    ext = Path(up.name.lower()).suffix
    buf = io.BytesIO(up.getvalue())
    if ext == ".csv":
        return read_csv(buf)
    if ext == ".xlsx":
        buf.seek(0)
        return pd.read_excel(buf, engine="openpyxl", dtype=str)
    raise ValueError(ext)

def to_m2(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)

def add_cols(df: pd.DataFrame, ref_i: int, m2_i: int, ref_lbl: str, m2_lbl: str) -> pd.DataFrame:
    out = df.iloc[:, [ref_i - 1, m2_i - 1]].copy()
    out.columns = [ref_lbl, m2_lbl]
    out[m2_lbl] = to_m2(out[m2_lbl])
    return out

def ram() -> str:
    if psutil is None:
        return "n/a"
    return f"{psutil.Process(os.getpid()).memory_info().rss/1_048_576:,.0f} Mo"

# ──────────── composant uploader ────────────

def uploader(prefix: str, lots: Dict[str, tuple[str, str, str]]):
    for k in lots:
        st.session_state.setdefault(f"{prefix}_{k}_files", [])
        st.session_state.setdefault(f"{prefix}_{k}_names", [])

    cols = st.columns(len(lots))
    for (k, (title, lab_ref, lab_val)), col in zip(lots.items(), cols):
        with col:
            st.subheader(title)
            ups = st.file_uploader("Deposer…", type=("csv", "xlsx"), accept_multiple_files=True, key=f"{prefix}_{k}_up")
            if ups:
                for up in ups:
                    if up.name not in st.session_state[f"{prefix}_{k}_names"]:
                        st.session_state[f"{prefix}_{k}_files"].append(up)
                        st.session_state[f"{prefix}_{k}_names"].append(up.name)
                st.success("Fichiers ajoutes.")
            st.number_input(lab_ref, 1, 50, 1, key=f"{prefix}_{k}_ref")
            st.number_input(lab_val, 1, 50, 2, key=f"{prefix}_{k}_val")
            st.caption(f"{len(st.session_state[f'{prefix}_{k}_files'])} fichier(s) | RAM: {ram()}")

# ──────────── PAGE PC ────────────
if page.startswith("Mise a jour M2 - PC"):
    st.header("Mise a jour des codes M2 (PC)")
    LOTS_PC = {
        "old": ("Donnees N-1", "Ref produit", "M2 ancien"),
        "new": ("Donnees N",   "Ref produit", "M2 nouveau"),
    }
    uploader("pc", LOTS_PC)

    if st.button("Generer M2_MisAJour"):
        if not all(st.session_state[f"pc_{k}_files"] for k in LOTS_PC):
            st.warning("Chargez les deux fichiers.")
            st.stop()
        dfs = {}
        for k in LOTS_PC:
            parts = [load_df(f) for f in st.session_state[f"pc_{k}_files"]]
            if any(p is None for p in parts):
                st.error("Lecture impossible.")
                st.stop()
            df = pd.concat(parts, ignore_index=True).drop_duplicates()
            if DEBUG_SAMPLE:
                df = df.head(DEBUG_SAMPLE)
            dfs[k] = df
        old_df = add_cols(dfs["old"], st.session_state["pc_old_ref"], st.session_state["pc_old_val"], "Ref", "M2_ancien")
        new_df = add_cols(dfs["new"], st.session_state["pc_new_ref"], st.session_state["pc_new_val"], "Ref", "M2_nouveau")
        merged = new_df.merge(old_df[["Ref", "M2_ancien"]], on="Ref", how="left")
        maj = merged.groupby("M2_nouveau")["M2_ancien"].agg(lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA).reset_index()
        st.session_state["pc_result"] = maj

    if "pc_result" in st.session_state:
        dstr = datetime.today().strftime("%y%m%d")
        st.download_button("Télécharger M2_MisAJour.csv", st.session_state["pc_result"].to_csv(index=False, sep=";"), file_name=f"M2_MisAJour_{dstr}.csv", mime="text/csv", key="dl_pc")
        st.dataframe(st.session_state["pc_result"].head())

# ──────────── PAGE Appairage ────────────
if page.startswith("Mise a jour M2 - Appairage"):
    st.header("Appairage M2 / famille client")
    LOTS_CL = {
        "old": ("Donnees N-1", "Ref produit", "M2 ancien"),
        "new": ("Donnees N",   "Ref produit", "M2 nouveau"),
        "map": ("Mapping",     "M2 ancien",   "Code famille client"),
    }
    uploader("cl", LOTS_CL)

    # colonne de menu déroulant supplémentaire
    add_cols_menu: List[str] = []

    if st.button("Generer appairage"):
        if not all(st.session_state[f"cl_{k}_files"] for k in LOTS_CL):
            st.warning("Chargez les 3 fichiers.")
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
        old_df = add_cols(dfs["old"], st.session_state["cl_old_ref"], st.session_state["cl_old_val"], "Ref", "M2_ancien")
        new_df = add_cols(dfs["new"], st.session_state["cl_new_ref"], st.session_state["cl_new_val"], "Ref", "M2_nouveau")
        map_df = dfs["map"].iloc[:, [st.session_state["cl_map_ref"]-1, st.session_state["cl_map_val"]-1]].copy()
        map_df.columns = ["M2_ancien", "Code_famille_Client"]
        map_df["M2_ancien"] = to_m2(map_df["M2_ancien"])
        old_df["M2_ancien"] = to_m2(old_df["M2_ancien"])
        merged = new_df.merge(old_df[["Ref", "M2_ancien"]], on="Ref", how="left").merge(map_df, on="M2_ancien", how="left")
        fam = merged.groupby("M2_nouveau")["Code_famille
