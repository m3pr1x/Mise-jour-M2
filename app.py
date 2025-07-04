# streamlit_app.py — Mise à jour M2 (PC) & Appairage client
from __future__ import annotations
import csv, io
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

try:                       # jauge RAM facultative
    import psutil
    RAM = lambda: f"{psutil.Process().memory_info().rss/1_048_576:,.0f} Mo"
except ModuleNotFoundError:
    psutil = None
    RAM = lambda: "n/a"    # type: ignore

TODAY = datetime.today().strftime("%y%m%d")
DEBUG_ROWS: int | None = None   # ex : 10000

# ---------------- I/O helpers ----------------
def read_csv(buf: io.BytesIO) -> pd.DataFrame:
    for enc in ("utf-8", "latin1", "cp1252"):
        buf.seek(0)
        try:
            sample = buf.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, encoding=enc, dtype=str,
                               low_memory=True, on_bad_lines="skip")
        except Exception:
            continue
    raise ValueError("CSV illisible")

@st.cache_data(show_spinner=False, hash_funcs={io.BytesIO: lambda _: None})
def load_df(up) -> pd.DataFrame:
    ext = Path(up.name.lower()).suffix
    b = io.BytesIO(up.getvalue())
    if ext == ".csv":
        return read_csv(b)
    if ext == ".xlsx":
        b.seek(0)
        return pd.read_excel(b, engine="openpyxl", dtype=str)
    raise ValueError(ext)

def to_m2(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)

def add_cols(df: pd.DataFrame, ref_i: int, m2_i: int,
             ref_label: str, m2_label: str) -> pd.DataFrame:
    sub = df.iloc[:, [ref_i-1, m2_i-1]].copy()
    sub.columns = [ref_label, m2_label]
    sub[m2_label] = to_m2(sub[m2_label])
    return sub

# ---------------- uploader component ----------------
def uploader(prefix: str, lots: Dict[str, tuple[str, str, str]]):
    for key in lots:
        st.session_state.setdefault(f"{prefix}_{key}_files", [])
        st.session_state.setdefault(f"{prefix}_{key}_names", [])

    cols = st.columns(len(lots))
    for (key, (title, lab_ref, lab_val)), col in zip(lots.items(), cols):
        with col:
            st.subheader(title)
            ups = st.file_uploader("Déposer...", type=("csv", "xlsx"),
                                   accept_multiple_files=True,
                                   key=f"{prefix}_{key}_up")
            if ups:
                for up in ups:
                    if up.name not in st.session_state[f"{prefix}_{key}_names"]:
                        st.session_state[f"{prefix}_{key}_files"].append(up)
                        st.session_state[f"{prefix}_{key}_names"].append(up.name)
                st.success("Upload ok")
            st.number_input(lab_ref, 1, 50, 1, key=f"{prefix}_{key}_ref")
            st.number_input(lab_val, 1, 50, 2, key=f"{prefix}_{key}_val")
            st.caption(f"{len(st.session_state[f'{prefix}_{key}_files'])} fichier(s) • RAM {RAM()}")

# ---------------- Streamlit layout ----------------
st.set_page_config(page_title="Mise a jour M2", page_icon="🛠", layout="wide")
page = st.sidebar.radio("Navigation",
                        ("Mise a jour M2 - PC", "Mise a jour M2 - Appairage client"))

# ══════════════════ PAGE PC ══════════════════
if page.startswith("Mise a jour M2 - PC"):
    st.header("Mise à jour des codes M2 (Personal Catalogue)")
    LOTS_PC = {
        "old": ("Donnees N-1", "Ref produit", "M2 ancien"),
        "new": ("Donnees N",   "Ref produit", "M2 nouveau"),
    }
    uploader("pc", LOTS_PC)

    if st.button("Générer M2_MisAJour"):
        if not all(st.session_state[f"pc_{k}_files"] for k in LOTS_PC):
            st.warning("Chargez N‑1 et N."); st.stop()

        dfs = {}
        for k in LOTS_PC:
            parts = [load_df(f) for f in st.session_state[f"pc_{k}_files"]]
            if any(p is None for p in parts): st.error("Erreur lecture"); st.stop()
            df = pd.concat(parts, ignore_index=True).drop_duplicates()
            if DEBUG_ROWS: df = df.head(DEBUG_ROWS)
            dfs[k] = df

        for k, df in dfs.items():
            ri, mi = st.session_state[f"pc_{k}_ref"], st.session_state[f"pc_{k}_val"]
            if not (1 <= ri <= df.shape[1] and 1 <= mi <= df.shape[1]):
                st.error(f"Indices hors plage ({k})"); st.stop()

        old_df = add_cols(dfs["old"], st.session_state["pc_old_ref"], st.session_state["pc_old_val"], "Ref", "M2_ancien")
        new_df = add_cols(dfs["new"], st.session_state["pc_new_ref"], st.session_state["pc_new_val"], "Ref", "M2_nouveau")

        merged = new_df.merge(old_df[["Ref", "M2_ancien"]], on="Ref", how="left")
        maj = merged.groupby("M2_nouveau")["M2_ancien"].agg(
            lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA
        ).reset_index()

        st.session_state["pc_csv"] = maj

    if "pc_csv" in st.session_state:
        st.download_button(
            "Télécharger M2_MisAJour.csv",
            st.session_state["pc_csv"].to_csv(index=False, sep=";"),
            file_name=f"M2_MisAJour_{TODAY}.csv",
            mime="text/csv",
            key="dl_pc",
        )
        st.dataframe(st.session_state["pc_csv"].head())

# ════════════════ PAGE APPAIRAGE ════════════════
if page.startswith("Mise a jour M2 - Appairage"):
    st.header("Appairage M2 / famille client")
    LOTS_CL = {
        "old": ("Donnees N-1", "Ref produit", "M2 ancien"),
        "new": ("Donnees N",   "Ref produit", "M2 nouveau"),
        "map": ("Mapping",     "M2 ancien",   "Code famille client"),
    }
    uploader("cl", LOTS_CL)

    extra_cols = st.multiselect(
        "Colonnes suppl. pour a_remplir.csv",
        options=st.session_state.get("cl_cols", []),
    )

    if st.button("Générer appairage"):
        if not all(st.session_state[f"cl_{k}_files"] for k in LOTS_CL):
            st.warning("Chargez les 3 fichiers."); st.stop()

        dfs = {}
        for k in LOTS_CL:
            parts = [load_df(f) for f in st.session_state[f"cl_{k}_files"]]
            if any(p is None for p in parts): st.error("Erreur lecture"); st.stop()
            df = pd.concat(parts, ignore_index=True).drop_duplicates()
            if DEBUG_ROWS: df = df.head(DEBUG_ROWS)
            dfs[k] = df

        for k, df in dfs.items():
            ri, vi = st.session_state[f"cl_{k}_ref"], st.session_state[f"cl_{k}_val"]
            if not (1 <= ri <= df.shape[1] and 1 <= vi <= df.shape[1]):
                st.error(f"Indices hors plage ({k})"); st.stop()

        old_df = add_cols(dfs["old"], st.session_state["cl_old_ref"], st.session_state["cl_old_val"], "Ref", "M2_ancien")
        new_df = add_cols(dfs["new"], st.session_state["cl_new_ref"], st.session_state["cl_new_val"], "Ref", "M2_nouveau")

        map_df = dfs["map"].iloc[:, [st.session_state["cl_map_ref"]-1, st.session_state["cl_map_val"]-1]].copy()
        map_df.columns = ["M2_ancien", "Code_famille_Client"]
        map_df["M2_ancien"] = to_m2(map_df["M2_ancien"])
        old_df["M2_ancien"] = to_m2(old_df["M2_ancien"])

        merged = (
            new_df.merge(old_df[["Ref", "M2_ancien"]], on="Ref", how="left")
            .merge(map_df, on="M2_ancien", how="left")
        )

        st.session_state["cl_cols"] = list(merged.columns)  # pour le multiselect

        fam = merged.groupby("M2_nouveau")["Code_famille_Client"].agg(
            lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA
        ).reset_index()

        a_remplir = fam[fam["Code_famille_Client"].isna()].copy()
        if extra_cols:
            cols_ok = [c for c in extra_cols if c in merged.columns]
            a_remplir = a_remplir.merge(merged[["M2_nouveau"] + cols_ok].drop_duplicates(), on="M2_nouveau", how="left")

        st.session_state["cl_csv_appair"] = fam
        st.session_state["cl_csv_missing"] = a_remplir

    if "cl_csv_appair" in st.session_state:
        st.download_button(
            "Télécharger appairage_M2_famille.csv",
            st.session_state["cl_csv_appair"].to_csv(index=False, sep=";"),
            file_name=f"appairage_M2_CodeFamilleClient_{TODAY}.csv",
            mime="text/csv",
            key="dl_app",
        )
        st.dataframe(st.session_state["cl_csv_appair"].head())
    if "cl_csv_missing" in st.session_state:
        st.download_button(
            "Télécharger a_remplir.csv",
            st.session_state["cl_csv_missing"].to_csv(index=False, sep=";"),
            file
