# ────────────────────────────────────────────────────────────
# fichier : appairage_m2_streamlit.py   (refonte 2025‑07‑04)
# ────────────────────────────────────────────────────────────
"""Streamlit : appairage codes M2 ⇆ familles client

• Lecture CSV/Excel robuste et **mise en cache** : première lecture → parse,
  reruns suivants → DataFrame directement récupéré (plus de roue « en cours… »).
• On ne stocke plus le flux binaire en mémoire ; on garde simplement
  l’`UploadedFile` (objet Streamlit) + son nom pour éviter les doublons.
• Limiteur d’échantillon facultatif pour le débogage (`SAMPLE_LIM` = nombre de
  lignes max par lot ; mettre `None` pour désactiver).
"""
from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

# ═════════════════════════  PARAMÈTRES  ═════════════════════════
SAMPLE_LIM: int | None = None   # ← mettre 10_000 en debug, None en prod

# ═════════════════════════  CONFIG UI  ═════════════════════════
st.set_page_config("Appairage M2", "🛠️", layout="wide")
st.title("🛠️ Appairage codes M2 / familles client")

# ═════════════════════  FONCTIONS UTILITAIRES  ═══════════════════

def read_csv_buf(buf: io.BytesIO) -> pd.DataFrame:
    """Détecte automatiquement encodage (utf‑8 / latin1 / cp1252) + séparateur."""
    for enc in ("utf-8", "latin1", "cp1252"):
        buf.seek(0)
        try:
            sample = buf.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
        except (csv.Error, UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("CSV illisible : encodage ou séparateur non reconnu.")


def parse_uploaded(file) -> pd.DataFrame:
    """Lit un UploadedFile Streamlit en DataFrame (CSV / Excel / Parquet)."""
    suffix = Path(file.name.lower()).suffix
    if suffix == ".csv":
        return read_csv_buf(io.BytesIO(file.getvalue()))
    if suffix in {".xlsx", ".xls"}:
        file.seek(0)
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        return pd.read_excel(file, engine=engine)
    if suffix == ".parquet":
        file.seek(0)
        return pd.read_parquet(file)
    raise ValueError(f"Extension non prise en charge : {suffix}")


# mise en cache — clé = hash du nom + taille + mtime
@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame | None:
    try:
        return parse_uploaded(file)
    except Exception as err:
        st.error(f"{file.name} : {err}")
        return None


def to_m2(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)


def add_cols(df: pd.DataFrame, ref_idx: int, m2_idx: int, ref_label: str, m2_label: str) -> pd.DataFrame:
    out = df.copy()
    out[ref_label] = out.iloc[:, ref_idx - 1].astype(str)
    out[m2_label] = to_m2(out.iloc[:, m2_idx - 1])
    return out


# ═══════════════════  INTERFACE DE TÉLÉVERSEMENT  ════════════════════
LOTS = {
    "old": ("Données N‑1", "Idx Réf. client", "Idx M2 ancien"),
    "new": ("Données N",   "Idx Réf. client", "Idx M2 nouveau"),
    "map": ("Table appairage", "Idx M2 ancien", "Idx Code famille client"),
}

for k in LOTS:
    st.session_state.setdefault(f"{k}_files", [])   # type: List[st.runtime.uploaded_file_manager.UploadedFile]
    st.session_state.setdefault(f"{k}_names", [])

cols = st.columns(3)
for (key, (title, lab_ref, lab_val)), col in zip(LOTS.items(), cols):
    with col:
        st.subheader(title)
        uploads = st.file_uploader("Ajouter fichier(s)…", accept_multiple_files=True,
                                   type=("csv", "xlsx", "xls", "parquet"), key=f"uploader_{key}")
        if uploads:
            new = 0
            for up in uploads:
                if up.name not in st.session_state[f"{key}_names"]:
                    st.session_state[f"{key}_files"].append(up)
                    st.session_state[f"{key}_names"].append(up.name)
                    new += 1
            if new:
                st.success(f"{new} nouveau(x) fichier(s) accepté(s)")

        st.number_input(lab_ref, 1, 50, 1, key=f"{key}_ref")
        st.number_input(lab_val, 1, 50, 2, key=f"{key}_val")
        st.caption(f"{len(st.session_state[f'{key}_files'])} fichier(s) chargé(s)")

# ════════════════════════  TRAITEMENT  ════════════════════════
if st.button("🔗 Lancer l'appairage"):
    if not all(st.session_state[f"{k}_files"] for k in LOTS):
        st.warning("Merci de charger les trois lots avant de continuer.")
        st.stop()

    # ---------- Chargement + concat ----------
    dfs: dict[str, list[pd.DataFrame]] = {}
    for key in LOTS:
        with st.spinner(f"Lecture des fichiers {key.upper()}…"):
            dfs[key] = [load_df(f) for f in st.session_state[f"{key}_files"]]
        if any(df is None for df in dfs[key]):
            st.error("Erreur dans au moins un fichier — corrigez puis relancez.")
            st.stop()

    old_raw = pd.concat(dfs["old"], ignore_index=True).drop_duplicates()
    new_raw = pd.concat(dfs["new"], ignore_index=True).drop_duplicates()
    map_raw = pd.concat(dfs["map"], ignore_index=True).drop_duplicates()

    # échantillon pour debug
    if SAMPLE_LIM:
        old_raw = old_raw.head(SAMPLE_LIM)
        new_raw = new_raw.head(SAMPLE_LIM)
        map_raw = map_raw.head(SAMPLE_LIM)

    # ---------- Vérif index ----------
    for df, key in ((old_raw, "old"), (new_raw, "new"), (map_raw, "map")):
        ref_i, val_i = st.session_state[f"{key}_ref"], st.session_state[f"{key}_val"]
        if not (1 <= ref_i <= df.shape[1] and 1 <= val_i <= df.shape[1]):
            st.error(f"Index hors plage dans le lot {key.upper()}.")
            st.stop()

    # ---------- Normalisation ----------
    old_df = add_cols(old_raw, st.session_state["old_ref"], st.session_state["old_val"], "Reference", "M2_ancien")
    new_df = add_cols(new_raw, st.session_state["new_ref"], st.session_state["new_val"], "Reference", "M2_nouveau")

    map_df = map_raw.copy()
    map_df["M2_ancien"] = to_m2(map_df.iloc[:, st.session_state["map_ref"] - 1])
    map_df["Code_famille_Client"] = map_df.iloc[:, st.session_state["map_val"] - 1].astype(str)
    map_df = map_df[["M2_ancien", "Code_famille_Client"]]

    # ---------- Fusions ----------
    with st.spinner("Fusion des lots…"):
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
    st.download_button("⬇️ Appairage M2 → Famille", appairage_df.to_csv(index=False, sep=";"),
                       file_name=f"appairage_M2_CodeFamilleClient_{dstr}.csv", mime="text/csv")
    st.download_button("⬇️ Mise à jour M2", m2_update_df.to_csv(index=False, sep=";"),
                       file_name=f"M2_MisAJour_{dstr}.csv", mime="text/csv")

    st.markdown(f"**{len(merged):,}** lignes fusionnées — codes famille renseignés pour **{appairage_df['Code_famille_Client'].notna().sum():,}** M2_nouveau")
