# ───────────────────────────────────────────────────────────
# streamlit_app.py     ← nom reconnu par Streamlit Cloud
# ───────────────────────────────────────────────────────────
"""
Appairage codes M2 ⇆ familles client
------------------------------------
• CSV / XLSX robustes, lecture *lazy* (dtype=str, low_memory=True).
• Cache DataFrame -> redémarrage instantané après un 1ᵉʳ run.
• Barre d’état détaillée : tu sais exactement où ça charge.
"""

from __future__ import annotations
import csv, io, os, psutil
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# ─────────────────────────  Réglages rapides  ─────────────────────────
DEBUG_SAMPLE_ROWS: int | None = None   # 10_000 pour debug, None sinon

# ─────────────────────────  Mise en page  ────────────────────────────
st.set_page_config(page_title="Appairage M2", page_icon="🛠️", layout="wide")
st.title("🛠️ Appairage codes M2 / familles client")

# ─────────────────────────  Helpers I/O  ─────────────────────────────
def _read_csv(buffer: io.BytesIO) -> pd.DataFrame:
    """Détecte encodage + séparateur, lit le CSV en dtype=str."""
    for enc in ("utf-8", "latin1", "cp1252"):
        buffer.seek(0)
        try:
            sample = buffer.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buffer.seek(0)
            return pd.read_csv(
                buffer, sep=sep, encoding=enc, dtype=str,
                low_memory=True, engine="python", on_bad_lines="skip"
            )
        except (csv.Error, UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("Impossible de détecter l'encodage / séparateur.")

@st.cache_data(show_spinner=False, hash_funcs={io.BytesIO: lambda _: None})
def load_df(upload) -> pd.DataFrame:
    """Charge un UploadedFile → DataFrame, résultat mis en cache."""
    suffix = Path(upload.name.lower()).suffix
    data = upload.getvalue()          # bytes immuables (clé du cache)
    buf = io.BytesIO(data)

    if suffix == ".csv":
        return _read_csv(buf)

    if suffix == ".xlsx":
        buf.seek(0)
        return pd.read_excel(buf, engine="openpyxl", dtype=str)

    raise ValueError(f"Extension non prise en charge : {suffix}")

def to_m2(series: pd.Series) -> pd.Series:
    """Formate la série en chaîne à 6 chiffres (zero‑padding)."""
    return series.astype(str).str.zfill(6)

def add_cols(df: pd.DataFrame, ref_idx: int, m2_idx: int,
             ref_name: str, m2_name: str) -> pd.DataFrame:
    """Ajoute/normalise colonne Réf. + colonne M2 n°."""
    out = df.iloc[:, [ref_idx-1, m2_idx-1]].copy()
    out.columns = [ref_name, m2_name]
    out[m2_name] = to_m2(out[m2_name])
    return out

def ram_usage() -> str:
    rss = psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    return f"{rss:,.0f} Mo"

# ─────────────────────────  Upload interface  ─────────────────────────
LOTS = {
    "old": ("Données N‑1",   "Réf. client",   "M2 ancien"),
    "new": ("Données N",     "Réf. client",   "M2 nouveau"),
    "map": ("Table mapping", "M2 ancien",     "Code famille client"),
}

for key in LOTS:
    st.session_state.setdefault(f"{key}_files", [])   # type: List[st.UploadedFile]
    st.session_state.setdefault(f"{key}_names", [])

cols = st.columns(3)
for (key, (title, lab_ref, lab_val)), col in zip(LOTS.items(), cols):
    with col:
        st.subheader(title)
        uploads = st.file_uploader(
            "Dépose fichier(s)…", type=("csv", "xlsx"),
            accept_multiple_files=True, key=f"uploader_{key}"
        )
        if uploads:
            fresh = 0
            for up in uploads:
                if up.name not in st.session_state[f"{key}_names"]:
                    st.session_state[f"{key}_files"].append(up)
                    st.session_state[f"{key}_names"].append(up.name)
                    fresh += 1
            if fresh:
                st.success(f"{fresh} fichier(s) ajouté(s)")

        st.number_input(lab_ref, 1, 50, 1, key=f"{key}_ref")
        st.number_input(lab_val, 1, 50, 2, key=f"{key}_val")
        st.caption(f"{len(st.session_state[f'{key}_files'])} fichier(s) | RAM : {ram_usage()}")

# ─────────────────────────  Traitement  ───────────────────────────────
if st.button("🔗  Lancer l’appairage"):
    if not all(st.session_state[f"{k}_files"] for k in LOTS):
        st.warning("Merci de charger les trois lots avant de continuer.")
        st.stop()

    dfs: dict[str, pd.DataFrame] = {}
    for key in LOTS:
        with st.status(f"Lecture & concat {key.upper()}…", expanded=False):
            parts = [load_df(up) for up in st.session_state[f"{key}_files"]]
            if any(df is None for df in parts):
                st.error("Lecture impossible dans un fichier.")
                st.stop()
            df_cat = pd.concat(parts, ignore_index=True).drop_duplicates()
            if DEBUG_SAMPLE_ROWS:
                df_cat = df_cat.head(DEBUG_SAMPLE_ROWS)
            dfs[key] = df_cat
            st.write(f"→ {len(df_cat):,} lignes (RAM : {ram_usage()})")

    # Vérification des index entrés
    for key, df in dfs.items():
        ref_i, val_i = st.session_state[f"{key}_ref"], st.session_state[f"{key}_val"]
        if not (1 <= ref_i <= df.shape[1] and 1 <= val_i <= df.shape[1]):
            st.error(f"Indice hors plage pour le lot {key.upper()}.")
            st.stop()

    # Pré‑formatage
    old_df = add_cols(dfs["old"], st.session_state["old_ref"], st.session_state["old_val"],
                      "Reference", "M2_ancien")
    new_df = add_cols(dfs["new"], st.session_state["new_ref"], st.session_state["new_val"],
                      "Reference", "M2_nouveau")

    map_df = dfs["map"].iloc[:, [st.session_state["map_ref"]-1,
                                 st.session_state["map_val"]-1]].copy()
    map_df.columns = ["M2_ancien", "Code_famille_Client"]
    map_df["M2_ancien"] = to_m2(map_df["M2_ancien"])

    # Fusion
    with st.status("Fusion…", expanded=False):
        merged = new_df.merge(old_df[["Reference", "M2_ancien"]], on="Reference", how="outer")
        merged = merged.merge(map_df, on="M2_ancien", how="left")
        st.write(f"Fusion OK : {len(merged):,} lignes (RAM : {ram_usage()})")

    st.dataframe(merged.head())

    # Appairages par majorité
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

    dstr = datetime.today().strftime("%y%m%d")
    st.download_button(
        "⬇️ Appairage M2 → Famille",
        family_map.to_csv(index=False, sep=";"),
        file_name=f"appairage_M2_CodeFamilleClient_{dstr}.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Mise à jour M2",
        m2_map[["M2_ancien", "M2_nouveau"]].to_csv(index=False, sep=";"),
        file_name=f"M2_MisAJour_{dstr}.csv",
        mime="text/csv",
    )

    st.success("✓ Fichiers prêts au téléchargement.")
