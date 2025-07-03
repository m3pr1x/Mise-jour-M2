# ────────────────────────────────────────────────────────────
# fichier : mise_a_jour_m2.py
# ────────────────────────────────────────────────────────────
from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------- Page ----------
st.set_page_config("Mise à jour M2", "🛠️", layout="wide")
st.title("🛠️  Mise à jour M2")

# ---------- Helpers ----------
def read_csv(buf: io.BytesIO) -> pd.DataFrame:
    """Lecture robuste d'un CSV : essai de plusieurs encodages + détecteur de séparateur."""
    for enc in ("utf-8", "latin1", "cp1252"):
        buf.seek(0)
        try:
            sample = buf.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
        except (UnicodeDecodeError, csv.Error, pd.errors.ParserError):
            continue
    raise ValueError("Fichier CSV illisible")

def read_any(upload) -> pd.DataFrame | None:
    """Retourne un DataFrame depuis un UploadFile (csv / xlsx / xls)."""
    name = upload.name.lower()
    suffix = Path(name).suffix
    if suffix == ".csv":
        return read_csv(upload)
    if suffix == ".xlsx":
        return pd.read_excel(upload, engine="openpyxl")
    if suffix == ".xls":
        try:
            return pd.read_excel(upload, engine="xlrd")
        except ImportError:
            st.error(
                "Le format .xls nécessite le paquet `xlrd<2.0.0`.\n"
                "Installe‑le puis relance l'application."
            )
            return None
    st.error(f"Extension non prise en charge : {suffix}")
    return None

def to_m2_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)

def add_cols(df: pd.DataFrame, ref_idx: int, m2_idx: int, label: str) -> pd.DataFrame:
    out = df.copy()
    out["RéférenceProduit"] = out.iloc[:, ref_idx - 1].astype(str)
    out[label] = to_m2_series(out.iloc[:, m2_idx - 1])
    return out

def safe_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Outer‑merge en évitant la collision de colonnes homonymes (hors clé)."""
    dup = {c: f"{c}_right" for c in right.columns if c in left.columns and c != "RéférenceProduit"}
    return left.merge(right.rename(columns=dup), on="RéférenceProduit", how="outer")

def build_final(df: pd.DataFrame, ent: str) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "M2": df["M2_nouveau"],
                "Entreprise": ent,
                "Code_famille_Client": df["Code_famille_Client"],
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

def idx_ok(df: pd.DataFrame, idx: int) -> bool:
    return 1 <= idx <= df.shape[1]

# ---------- Configuration des 3 lots ----------
LOTS = {
    "cat": ("Catalogue interne",  "Idx Réf. produit", "Idx M2 actuelle"),
    "hist":("Historique ventes",  "Idx Réf. produit", "Idx M2 dernière"),
    "cli": ("Fichier client",     "Idx M2",           "Idx Code famille"),
}

for key in LOTS:
    st.session_state.setdefault(f"{key}_files", [])   # liste d'UploadedFile déjà ajoutés
    st.session_state.setdefault(f"{key}_names", [])   # juste les noms, pour éviter les doublons

cols = st.columns(3)
for (key, (title, lab_ref, lab_val)), col in zip(LOTS.items(), cols):
    with col:
        st.subheader(title)
        uploads = st.file_uploader(
            "Glisser‑déposer / parcourir…",
            accept_multiple_files=True,
            type=("csv", "xlsx", "xls"),
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

entreprise = st.text_input("Entreprise (en MAJUSCULES)").strip().upper()

# ---------- Fusion / appairage ----------
if st.button("🔗  Créer l'appairage M2 ➜ Code client"):
    # Vérifications
    if not all(st.session_state[f"{k}_files"] for k in LOTS):
        st.warning("Charge d'abord les trois lots de données.")
        st.stop()
    if not entreprise:
        st.warning("Renseigne le champ « Entreprise ».")
        st.stop()

    # Lecture des fichiers → DataFrames (on n'en garde pas la trace en session)
    dfs = {}
    for key in LOTS:
        dfs[key] = [read_any(up) for up in st.session_state[f"{key}_files"]]
        if any(df is None for df in dfs[key]):
            st.error("Erreur de lecture dans un des fichiers ; corrige puis réessaie.")
            st.stop()

    cat_raw  = pd.concat(dfs["cat"],  ignore_index=True).drop_duplicates()
    hist_raw = pd.concat(dfs["hist"], ignore_index=True).drop_duplicates()
    cli_raw  = pd.concat(dfs["cli"],  ignore_index=True).drop_duplicates()

    # Vérif des index
    for df, key in ((cat_raw, "cat"), (hist_raw, "hist"), (cli_raw, "cli")):
        if not idx_ok(df, st.session_state[f"{key}_ref"]) or not idx_ok(df, st.session_state[f"{key}_val"]):
            st.error(f"Index hors limites pour le lot {key.upper()}.")
            st.stop()

    # Pré‑traitement
    cat  = add_cols(cat_raw,  st.session_state["cat_ref"],  st.session_state["cat_val"],  "M2_nouveau")
    hist = add_cols(hist_raw, st.session_state["hist_ref"], st.session_state["hist_val"], "M2_ancien")

    cli_m2 = cli_raw.copy()
    cli_m2["M2"] = to_m2_series(cli_m2.iloc[:, st.session_state["cli_ref"] - 1])
    cli_m2["Code_famille_Client"] = cli_m2.iloc[:, st.session_state["cli_val"] - 1].astype(str)
    cli_m2 = cli_m2[["M2", "Code_famille_Client"]]

    with st.spinner("Fusion des fichiers…"):
        merged = safe_merge(cat, hist[["RéférenceProduit", "M2_ancien"]])
        merged = merged.merge(
            cli_m2,
            left_on="M2_ancien",
            right_on="M2",
            how="left",
            suffixes=("_cat", ""),
        )
        if "M2_cat" in merged.columns:
            merged.drop(columns=["M2_cat"], inplace=True)

    # Complétion majoritaire
    pre_assigned = merged["Code_famille_Client"].notna().sum()
    freq = (
        merged.dropna(subset=["Code_famille_Client"])
        .groupby("M2_nouveau")["Code_famille_Client"]
        .agg(lambda s: s.value_counts().idxmax())
    )

    merged["Code_famille_Client"] = merged.apply(
        lambda r: freq.get(r["M2_nouveau"], pd.NA)
        if pd.isna(r["Code_famille_Client"])
        else r["Code_famille_Client"],
        axis=1,
    )
    completed = merged["Code_famille_Client"].notna().sum() - pre_assigned

    # Résumé texte
    maj_list = [f"{m2} -> {code}" for m2, code in freq.items()]
    missing_final = merged[merged["Code_famille_Client"].isna()]["M2_nouveau"].unique()

    summary_txt = "\n".join(
        [
            f"M2 déjà codés : {pre_assigned}",
            f"M2 complétés (majorité) : {completed}",
            "",
            "M2 ajoutés / code choisi :",
            *maj_list,
            "",
            "M2 restants sans code :",
            *missing_final.astype(str),
        ]
    )

    # DataFrame final
    final_df = build_final(merged.drop_duplicates("M2_nouveau"), entreprise)
    dstr = datetime.today().strftime("%y%m%d")

    st.success("✅  Appairage terminé")
    st.dataframe(final_df.head())

    st.download_button(
        "⬇️ Télécharger l'appairage (CSV)",
        final_df.to_csv(index=False, sep=";"),
        file_name=f"APPARIAGE_M2_{entreprise}_{dstr}.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Télécharger le rapport (TXT)",
        summary_txt,
        file_name=f"SUIVI_{entreprise}_{dstr}.txt",
        mime="text/plain",
    )
