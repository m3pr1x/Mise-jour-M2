# ────────────────────────────────────────────────────────────
# fichier : mise_a_jour_m2.py
# ────────────────────────────────────────────────────────────
from __future__ import annotations
import csv, io
from datetime import datetime

import pandas as pd
import streamlit as st

# ---------- Page ----------
st.set_page_config("Mise à jour M2", "🛠️", layout="wide")
st.title("🛠️ Mise à jour M2")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def read_csv(buf: io.BytesIO) -> pd.DataFrame:
    """Essaye successivement plusieurs encodages + séparateurs."""
    for enc in ("utf-8", "latin1", "cp1252"):
        buf.seek(0)
        try:
            sample = buf.read(2048).decode(enc, errors="ignore")
            sep = csv.Sniffer().sniff(sample, delimiters=";,|\t").delimiter
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, encoding=enc, engine="python", on_bad_lines="skip")
        except (UnicodeDecodeError, csv.Error, pd.errors.ParserError):
            continue
    raise ValueError("Impossible de lire le fichier")

@st.cache_data(show_spinner=False)
def read_any(upload) -> pd.DataFrame | None:
    name = upload.name.lower()
    if name.endswith(".csv"):
        return read_csv(upload)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload, engine="openpyxl")
    return None

@st.cache_data(show_spinner=False)
def concat_unique(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True) if dfs else pd.DataFrame()

def to_m2_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(6)

def add_cols(df: pd.DataFrame, ref_idx: int, m2_idx: int, label: str) -> pd.DataFrame:
    out = df.copy()
    out["RéférenceProduit"] = out.iloc[:, ref_idx - 1].astype(str)
    out[label] = to_m2_series(out.iloc[:, m2_idx - 1])
    return out

def safe_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Outer‑merge en évitant de perdre des colonnes homonymes."""
    dup = {c: f"{c}_right" for c in right.columns if c in left.columns and c != "RéférenceProduit"}
    return left.merge(right.rename(columns=dup), on="RéférenceProduit", how="outer")

def build_final(df: pd.DataFrame, ent: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "M2": df["M2_nouveau"],
            "Entreprise": ent,
            "Code_famille_Client": df["Code_famille_Client"],
        }
    ).drop_duplicates()

# ---------- Configuration des 3 lots ----------
LOTS = {
    "cat": ("Catalogue interne",  "Idx Réf. produit", "Idx M2 actuelle"),
    "hist":("Historique ventes",  "Idx Réf. produit", "Idx M2 dernière"),
    "cli": ("Fichier client",     "Idx M2",           "Idx Code famille"),
}
for k in LOTS:
    st.session_state.setdefault(f"{k}_dfs", [])
    st.session_state.setdefault(f"{k}_names", [])

cols = st.columns(3)
for (k, (titre, lab_ref, lab_val)), col in zip(LOTS.items(), cols):
    with col:
        st.subheader(titre)
        uploads = st.file_uploader("Glisse‑dépose / browse…", accept_multiple_files=True,
                                   type=("csv", "xlsx", "xls"), key=f"up_{k}")
        if uploads:
            for up in uploads:
                if up.name not in st.session_state[f"{k}_names"]:
                    df = read_any(up)
                    if df is not None:
                        st.session_state[f"{k}_dfs"].append(df)
                        st.session_state[f"{k}_names"].append(up.name)
            st.success(f"{len(uploads)} fichier(s) ajouté(s)")
        st.number_input(lab_ref, 1, 50, 1, key=f"{k}_ref")
        st.number_input(lab_val, 1, 50, 2, key=f"{k}_val")
        st.caption(f"{len(st.session_state[f'{k}_dfs'])} fichier(s) chargés")

entreprise = st.text_input("Entreprise (MAJUSCULES)").strip().upper()

# ---------- Fusion / appairage ----------
def idx_ok(df: pd.DataFrame, idx: int) -> bool:
    return 1 <= idx <= df.shape[1]

if st.button("🔗 Créer l'appairage M2 ➜ Code client"):
    # Vérifications de base
    if not all(st.session_state[f"{k}_dfs"] for k in LOTS):
        st.warning("Il faut charger les 3 lots de données."); st.stop()
    if not entreprise:
        st.warning("Merci de renseigner l'entreprise."); st.stop()

    cat_raw  = concat_unique(st.session_state["cat_dfs"])
    hist_raw = concat_unique(st.session_state["hist_dfs"])
    cli_raw  = concat_unique(st.session_state["cli_dfs"])

    for df, key in ((cat_raw, "cat"), (hist_raw, "hist"), (cli_raw, "cli")):
        if not idx_ok(df, st.session_state[f"{key}_ref"]) or not idx_ok(df, st.session_state[f"{key}_val"]):
            st.error(f"Index hors limites pour le lot {key.upper()}"); st.stop()

    # Pré‑traitement
    cat  = add_cols(cat_raw,  st.session_state["cat_ref"],  st.session_state["cat_val"],  "M2_nouveau")
    hist = add_cols(hist_raw, st.session_state["hist_ref"], st.session_state["hist_val"], "M2_ancien")

    cli_m2 = cli_raw.copy()
    cli_m2["M2"] = to_m2_series(cli_m2.iloc[:, st.session_state["cli_ref"] - 1])
    cli_m2["Code_famille_Client"] = cli_m2.iloc[:, st.session_state["cli_val"] - 1].astype(str)
    cli_m2 = cli_m2[["M2", "Code_famille_Client"]]

    # Fusion n°1 : catalogue + historique
    merged = safe_merge(cat, hist[["RéférenceProduit", "M2_ancien"]])

    # Fusion n°2 : + fichier client
    merged = merged.merge(
        cli_m2,
        left_on="M2_ancien",
        right_on="M2",
        how="left",
        suffixes=("_cat", "")  # évite le doublon "M2"
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
        lambda r: freq.get(r["M2_nouveau"], pd.NA) if pd.isna(r["Code_famille_Client"]) else r["Code_famille_Client"],
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

    st.subheader("✅ Appairage M2 ➜ Code client")
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
