# ───────────────────────────────────────────────────────────
# streamlit_app.py  –  version « no‑psutil » pour Streamlit Cloud
# ───────────────────────────────────────────────────────────
"""Appairage codes M2 ⇆ familles client (RAM‑friendly).

Différence vs la version précédente
===================================
• **psutil devient optionnel** : s’il n’est pas installé, l’app tourne quand
  même (on masque simplement l’indicateur RAM). Ainsi, plus d’erreur
  « ModuleNotFoundError: psutil » sur Streamlit Cloud sans modifier
  `requirements.txt`.

• Pour avoir la jauge mémoire, ajoute `psutil` dans `requirements.txt` (facultatif).
"""
from __future__ import annotations
import csv, io, os
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

try:
    import psutil  # facultatif
except ModuleNotFoundError:  # Streamlit Cloud sans psutil
    psutil = None

DEBUG_SAMPLE_ROWS: int | None = None   # 10 000 pour debug

st.set_page_config(page_title="Appairage M2", page_icon="🛠️", layout="wide")
st.title("🛠️ Appairage codes M2 / familles client")

# ───────────────  Helpers  ───────────────

def _read_csv(buffer: io.BytesIO) -> pd.DataFrame:
    for enc in ("utf-8", "latin1", "cp1252"):
        buffer.seek(0)
        try:
            sample = buffer.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buffer.seek(0)
            return pd.read_csv(buffer, sep=sep, encoding=enc, dtype=str,
                               low_memory=True, engine="python", on_bad_lines="skip")
        except (csv.Error, UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("CSV illisible : encodage/séparateur non détecté.")

@st.cache_data(show_spinner=False, hash_funcs={io.BytesIO: lambda _: None})
def load_df(file) -> pd.DataFrame:
    ext = Path(file.name.lower()).suffix
    data = file.getvalue()
    buf = io.BytesIO(data)
    if ext == ".csv":
        return _read_csv(buf)
    if ext == ".xlsx":
        buf.seek(0)
        return pd.read_excel(buf, engine="openpyxl", dtype=str)
    raise ValueError(f"Extension {ext} non prise en charge")


def to_m2(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)


def add_cols(df: pd.DataFrame, ref_idx: int, m2_idx: int,
             ref_name: str, m2_name: str) -> pd.DataFrame:
    out = df.iloc[:, [ref_idx-1, m2_idx-1]].copy()
    out.columns = [ref_name, m2_name]
    out[m2_name] = to_m2(out[m2_name])
    return out


def ram_usage() -> str:
    if psutil is None:
        return "n/a"
    mem = psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    return f"{mem:,.0f} Mo"

# ───────────────  Upload UI  ───────────────
LOTS = {
    "old": ("Données N‑1",   "Réf. client", "M2 ancien"),
    "new": ("Données N",     "Réf. client", "M2 nouveau"),
    "map": ("Table mapping", "M2 ancien",   "Code famille client"),
}
for k in LOTS:
    st.session_state.setdefault(f"{k}_files", [])   # list[UploadedFile]
    st.session_state.setdefault(f"{k}_names", [])

cols = st.columns(3)
for (key, (title, lab_ref, lab_val)), col in zip(LOTS.items(), cols):
    with col:
        st.subheader(title)
        uploads = st.file_uploader("Ajouter fichier(s)…", type=("csv", "xlsx"),
                                   accept_multiple_files=True, key=f"uploader_{key}")
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
        st.caption(f"{len(st.session_state[f'{key}_files'])} fichier(s) | RAM : {ram_usage()}")

# ───────────────  Traitement  ───────────────
if st.button("🔗 Lancer l'appairage"):
    if not all(st.session_state[f"{k}_files"] for k in LOTS):
        st.warning("Chargez les 3 lots avant de continuer.")
        st.stop()

    dfs: dict[str, pd.DataFrame] = {}
    for key in LOTS:
        with st.status(f"Lecture {key.upper()}…", expanded=False):
            parts = [load_df(f) for f in st.session_state[f"{key}_files"]]
            if any(p is None for p in parts):
                st.error("Erreur de lecture.")
                st.stop()
            df = pd.concat(parts, ignore_index=True).drop_duplicates()
            if DEBUG_SAMPLE_ROWS:
                df = df.head(DEBUG_SAMPLE_ROWS)
            dfs[key] = df
            st.write(f"{len(df):,} lignes (↯ {ram_usage()})")

    # vérif index
    for key, df in dfs.items():
        ref_i, val_i = st.session_state[f"{key}_ref"], st.session_state[f"{key}_val"]
        if not (1 <= ref_i <= df.shape[1] and 1 <= val_i <= df.shape[1]):
            st.error(f"Index hors plage sur {key.upper()}.")
            st.stop()

    # formatage
    old = add_cols(dfs["old"], st.session_state["old_ref"], st.session_state["old_val"], "Reference", "M2_ancien")
    new = add_cols(dfs["new"], st.session_state["new_ref"], st.session_state["new_val"], "Reference", "M2_nouveau")

    map_df = dfs["map"].iloc[:, [st.session_state["map_ref"]-1, st.session_state["map_val"]-1]].copy()
    map_df.columns = ["M2_ancien", "Code_famille_Client"]
    map_df["M2_ancien"] = to_m2(map_df["M2_ancien"])

    with st.status("Fusion…", expanded=False):
        merged = new.merge(old[["Reference", "M2_ancien"]], on="Reference", how="outer")
        merged = merged.merge(map_df, on="M2_ancien", how="left")
        st.write(f"Fusion OK ({len(merged):,} lignes)")

    # majorité
    fam = merged.groupby("M2_nouveau")["Code_famille_Client"].agg(lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA).reset_index()
    rel = merged.groupby("M2_nouveau")["M2_ancien"].agg(lambda s: s.value_counts().idxmax() if s.notna().any() else pd.NA).reset_index()

    dstr = datetime.today().strftime("%y%m%d")
    st.download_button("⬇️ Appairage", fam.to_csv(index=False, sep=";"), file_name=f"appairage_M2_CodeFamilleClient_{dstr}.csv", mime="text/csv")
    st.download_button("⬇️ M2 Mis à jour", rel.to_csv(index=False, sep=";"), file_name=f"M2_MisAJour_{dstr}.csv", mime="text/csv")

    st.success("Fichiers prêts au téléchargement.")
