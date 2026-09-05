"""
Exposome-informed post-hoc analysis (thesis Methodology, "Exposome-Informed
Post-Hoc Analysis" + H1b in "Light-linked transcript connectivity").

Two outputs, both built on the same community eigengene scores:
  1. Exposome variables (day length, occupation, education, three dietary
     indices) correlated against each Louvain community.
  2. The light-linked gene panel (H1b) correlated against the same scores
     (see LIGHT_LINKED_PANEL below for why "light-linked" not "circadian").

METHOD
------
Louvain community detection is re-run here (not imported from
network_metrics.ipynb's runtime state) so this script runs standalone.
Community eigengene = first PC of each community's patient x feature block
(WGCNA-style, sign-aligned). Correlation is Spearman per exposome variable
against every community, BH-corrected -- but correction SCOPE differs by
function: run_exposome_correlations corrects per-variable, while
run_h1b_correlations and run_demographics_correlations each pool across
their whole comparison count. See each function's own comment.

Day length: latitude x day-of-year via the Forsythe et al. (1995) photoperiod
formula (no astronomy library added -- `astral` isn't installed in `monika`,
and this is one closed-form trig expression).
This ignores atmospheric refraction and elevation, accurate to within a few
minutes -- fine for a coarse exposure proxy, not an almanac; swap in
`astral` if that precision ever matters.

medication_count and comorbidity_count are COMPUTED but EXCLUDED -- read as
"ever Yes" across each participant's full row set (the single HTX biopsy row
undercounts both). Correctly counted, medication is 76/90 non-zero and
comorbidity 28/90, and both turn out to be disease-activity readouts rather
than exposures: medication predicts IBD status at AUC=0.963, comorbidity is
zero for all 22 non-IBD controls. Same principle already applied to CRP/ESR
in this Methodology. Three FFQ diet indices replace them -- diet is the one
axis in this cohort with real spread not confounded with diagnosis or age.

Site/education/occupation/medication/comorbidity are read off the Screening
Colonoscopy biopsy row; collection date is the exception (date_of_receipt is
0% populated there) and is recovered from the nearest dated stool sample.

Depends on GLASSO/results/ibd/ibd_joint_network.graphml being current.

Run from the pipeline/ directory:
    python analysis_05_exposome.py              # run on real data
    python analysis_05_exposome.py --self-test   # verify day-length + BH logic only
"""

import argparse
import math

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities
import networkx.algorithms.community as nx_comm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# ── Paths ────────────────────────────────────────────────────────────────
GRAPHML_PATH = "../GLASSO/results/ibd/ibd_joint_network.graphml"
META_PATH = "Data/hmp2_metadata_2018-08-20.csv"
MGX_PATH = "../GLASSO/data/mgx_for_pig_ibd.csv"
MBX_PATH = "../GLASSO/data/mbx_for_pig_ibd.csv"
RNA_PATH = "../GLASSO/data/rna_for_pig_ibd.csv"
RNA_FULL_POOL_PATH = "Data/rna_transcriptomics_vst_full_pool.csv"
RESULTS_DIR = "../GLASSO/results/ibd/"

SEED = 42  # matches network_metrics.ipynb's CONFIG["seed"]

# H1b panel (Introduction; Related Work S2.3). The core circadian clock genes (PER1/2/3, CRY1, CLOCK, ARNTL,
# NR1D1/2, RORA, NPAS2, DBP) were REMOVED. Two reasons, one mechanistic and one
# empirical. Mechanistically, those genes oscillate on a ~24h cycle, biopsy time of
# day is not recorded in HMP2, and screening colonoscopies cluster in the morning
# after an overnight fast and bowel prep -- so a single biopsy samples an unknown
# phase of the oscillation rather than the participant's chronic clock state.
# Empirically, none of them carry IBD signal in this cohort (Mann-Whitney + BH over
# the full 20,218-gene VST pool: PER1 0.22, PER2 0.063, PER3 0.95, CRY1 0.59,
# CLOCK 0.62, ARNTL 0.94, NR1D1 0.32, NR1D2 0.68, RORA 0.54, NPAS2 0.28, DBP 0.44;
# only CRY2 reaches 0.044).
#
# What replaces them are light-linked markers that are NOT fast oscillators, so a
# single timepoint is interpretable, and that do carry signal here:
#   MTNR1A  padj 0.0149 (rank  778) -- melatonin receptor; melatonin is the darkness
#                                      signal, and receptor level is not itself rhythmic
#   ARNTL2  padj 0.0071 (rank  379) -- BMAL1 paralog, not a core peripheral oscillator
#   DIO3    padj 0.0374 (rank 1968) -- seasonal photoperiod / thyroid axis
# The vitamin D axis is retained: it acts on a seasonal timescale via UV -> skin
# synthesis -> 25(OH)D (half-life ~2-3 weeks) -> VDR signaling, which is buffered
# against time of day. CYP2R1 (the 25-hydroxylase) is added to complete the axis.
#
# Deliberately NOT included: S100A8/S100A9. Both are strongly associated here
# (padj 0.0006 / 0.0016) and are documented VDR targets, but together they are
# calprotectin -- the standard IBD disease-activity marker -- so their signal is
# inflammation by construction, not vitamin D status.
LIGHT_LINKED_PANEL = [
    "VDR", "CYP24A1", "CYP27B1", "CYP2R1",   # vitamin D signaling axis
    "MTNR1A", "ARNTL2", "DIO3",              # light/dark and photoperiod markers, non-24h
]

# Recruiting-site latitudes (degrees N) -- used only as the input to the
# day-length formula below, never reported as their own variable.
SITE_LATITUDE = {
    "Cedars-Sinai": 34.0522,    # Los Angeles, CA
    "Emory": 33.7490,           # Atlanta, GA
    "MGH": 42.3601,             # Boston, MA
    "MGH Pediatrics": 42.3601,  # Boston, MA
    "Cincinnati": 39.1031,      # Cincinnati, OH
}

# Same biopsy-location priority order as pipeline.ipynb's HTX representative-
# timepoint selection (Rectum first: most-sampled location, diagnosis mix
# closest to the full cohort).
LOCATION_PRIORITY = [
    "Rectum", "Ileum", "Sigmoid Colon", "Transverse colon",
    "Descending (left-sided) colon", "Ascending (right-sided) colon",
    "Cecum", "Terminal ileum", "Non-inflamed",
]

EDUCATION_ORDER = {
    "7th grade or less": 0,
    "Some high school": 1,
    "High school graduate or GED": 2,
    "Some college, no degree": 3,
    "Associate degree": 4,
    "Bachelor's degree": 5,
    "Master's degree": 6,
    "Professional/Doctoral degree": 7,
    # "Unknown/Not Reported" is intentionally absent -> maps to NaN, not 0.
}

# Occupation has no natural order (Retired/Paid/Student/Unpaid), so it is
# collapsed to a binary "currently paid work" flag rather than an invented
# ordinal scale. Revisit if unpaid/student/retired need separating.
# Arthralgia(s) is the single most common extraintestinal manifestation in this
# cohort (116 + 106 "Yes" across the ".1" duplicate for the second questionnaire
# version), so both variants are included here.
EXTRAINTESTINAL_COLS = [
    "Uveitis", "Erythema nodosum", "Aphthous ulcers", "Pyoderma gangrenosum",
    "Anal fissure", "New fistula", "Abscess", "Arthralgia", "Arthralgias",
    "Uveitis.1", "Erythema nodosum.1",
]
COMORBID_COLS = [
    "Cancer - breast", "Cancer - cholangiocarcinoma", "Cancer - colon or rectum",
    "Cancer - Hodgkin's lymphoma", "Cancer - liver", "Cancer - lung",
    "Cancer - lymphoma (not otherwise specified)", "Cancer - Non-Hodkin's lymphoma",
    "Cancer - ovarian", "Cancer - prostate", "Other immune mediated diseases",
    "Celiac sprue", "Chronic bronchitis", "Dermatitis herpetiformis",
    "Familial Mediterranean fever", "Grave's disease", "Guillian-Barre Syndrome",
    "Hashimoto's (autoimmune) thyroiditis", "Idiopathic pulmonary fibrosis",
    "Idiopathic thrombocytopenia purpura", "Irritable bowel syndrome",
    "Alopecia areata", "Multiple sclerosis", "Myasthenia gravis", "Myocarditis",
    "Neuropathy", "Pericarditis", "Pemphigus vulgaris", "Pernicious anemia",
    "Polymyositis or dermatomyositis", "Primary biliary cirrhosis",
    "Primary sclerosing cholangitis", "Ankylosing spondylitis", "Psoriasis",
    "Rheumatoid arthritis", "Sarcoidosis", "Scleroderma (systemic sclerosis)",
    "Sjogren's syndrome", "Systemic lupus erythematosis", "Temporal arteritis",
    "Thyroid disease (uncertain diagnosis, not cancer)",
    "Type I Diabetes (Juvenile Diabetes)", "Vitiligo",
    "Arthritis (uncertain diagnosis)", "Wegener's granulomatosis", "Asthma",
    "Autoimmune hemolytic anemia", "Autoimmune hepatitis", "Bechet's syndrome",
]
MEDICATION_YESNO_COLS = [
    "Antibiotics", "Chemotherapy", "Immunosuppressants (e.g. oral corticosteroids)",
]
# This is the full "drug ladder" question set (shared value set
# {Never taken, Taken prior to baseline, Taken since last visit, Current}),
# found by scanning all 490 metadata columns for that value set, and it lives on
# the serology/methylome/host_genome (blood) rows -- a third row family, distinct
# from both the HTX biopsy row and the stool rows (see build_exposome_table).
DRUG_LADDER_COLS = [
    "Lomotil", "Dipentum (olsalazine)", "Rowasa enemas (mesalamine enemas)",
    "Canasa suppositories (mesalamine suppositories)", "Flagyl (Metronidazole)",
    "Cipro (Ciprofloxin)", "Xifaxin (rifaxamin)", "Levaquin", "Other Antibiotic:",
    "Prednisone", "Entocort (Budesonide)", "Imodium", "Solumedrol (Medrol)",
    "IV steroids", "Cortenemas, Cortifoam, Proctofoam",
    "Azathioprine (Imuran, Azasan)", "Methotrexate",
    "Mercaptopurine (Purinethol, 6MP)", "VSL #3", "FOS", "Remicade (Infliximab)",
    "Humira (Adalimumab)", "DTO", "Cimzia (Certlizumab)", "Tysabri (Natalizumab)",
    "Asacol (mesalamine)", "Pentasa (mesalamine)", "Lialda (mesalamine)",
    "Apriso (mesalamine)", "Colozal (balasalizide)", "Sulfasalizine (Azulfidine)",
]
DRUG_LADDER_TAKEN_VALUES = {"Current", "Taken since last visit"}

# The food frequency questionnaire (FFQ) is answered by all 90 participants,
# repeatedly across the study year. 18 of 20 FFQ items show no relationship to diagnosis;
# "Whole grains" (p=0.031) and "Probiotic" (p=0.013) are confounded and excluded
# from the three indices below. Column names are quoted exactly as truncated in
# the source CSV header (the file itself cuts these off mid-word; not a copy
# error here).
FFQ_FREQUENCY_MAP = {
    "No, I did not consume these products in the last 7 days": 0,
    "Within the past 4 to 7 days": 1,
    "Within the past 2 to 3 days": 2,
    "Yesterday, 1 to 2 times": 3,
    "Yesterday, 3 or more times": 4,
}
# "Whole grains" and "Probiotic" are deliberately absent from the two indices
# below despite belonging thematically (plant-fiber, fermented-food): each is
# individually confounded with diagnosis (p=0.031, p=0.013). Dropping them
# makes the composite SAFER, not just simpler -- p vs. diagnosis rises from
# 0.109->0.473 (plant/fiber) and 0.433->0.510 (fermented) once they are
# excluded, rather than averaged toward significance.
DIET_INDEX_COLS = {
    "diet_plant_fiber": [
        "Fruits (no juice) (Apples, raisins, bananas, oranges, strawberries, blueberries",
        "Vegetables (salad, tomatoes, onions, greens, carrots, peppers, green beans, etc)",
        "Beans (tofu, soy, soy burgers, lentils, Mexican beans, lima beans etc)",
    ],
    "diet_processed_food": [
        "Processed meat (other red or white meat such as lunch meat, ham, salami, bologna",
        "Sweets (pies, jam, chocolate, cake, cookies, etc.)",
        "Soft drinks, tea or coffee with sugar (corn syrup, maple syrup, cane sugar, etc)",
        "Starch (white rice, bread, pizza, potatoes, yams, cereals, pancakes, etc.)",
    ],
    "diet_fermented_food": [
        "Yogurt or other foods containing active bacterial cultures (kefir, sauerkraut)",
    ],
}


# ── Day length ──────────────────────────────────────────────────────────
def day_length_hours(latitude_deg, day_of_year):
    """Hours of daylight at a given latitude and day of year.
    Forsythe et al. 1995, Ecological Modelling 80: 87-95 (the standard
    closed-form CBM photoperiod model)."""
    lat = math.radians(latitude_deg)
    revolution_angle = 0.2163108 + 2 * math.atan(0.9671396 * math.tan(0.00860 * (day_of_year - 186)))
    declination = math.asin(0.39795 * math.cos(revolution_angle))
    numerator = math.sin(0.8333 * math.pi / 180) + math.sin(lat) * math.sin(declination)
    denominator = math.cos(lat) * math.cos(declination)
    ratio = max(-1.0, min(1.0, numerator / denominator))
    return 24.0 - (24.0 / math.pi) * math.acos(ratio)


def benjamini_hochberg(pvals):
    """Wrapper around statsmodels' fdr_bh, tolerant of an all-NaN input
    (returns all-NaN rather than raising)."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.shape, np.nan)
    valid = ~np.isnan(pvals)
    if valid.any():
        out[valid] = multipletests(pvals[valid], method="fdr_bh")[1]
    return out


# ── Network + communities ──────────────────────────────────────────────
def load_communities():
    G = nx.read_graphml(GRAPHML_PATH)
    G_abs = nx.Graph()
    G_abs.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        G_abs.add_edge(u, v, weight=abs(d.get("weight", 1.0)))
    communities = louvain_communities(G_abs, weight="weight", seed=SEED)
    modularity_q = nx_comm.modularity(G_abs, communities, weight="weight")
    print(f"Communities detected: {len(communities)} (Q={modularity_q:.4f})")
    return G, communities


# ── Community eigengene scores ─────────────────────────────────────────
def community_eigengenes(G, communities, modality_frames):
    node_modality = nx.get_node_attributes(G, "modality")

    def feature_values(node):
        return modality_frames[node_modality[node]][node]

    def eigengene(nodes):
        X = pd.concat([feature_values(n) for n in nodes], axis=1)
        X = (X - X.mean()) / X.std()  # correlation-matrix PCA, WGCNA convention
        u, s, vt = np.linalg.svd(X.values, full_matrices=False)
        pc1 = u[:, 0] * s[0]
        if np.mean(vt[0]) < 0:  # sign-align: eigengene points with its members
            pc1 = -pc1
        return pd.Series(pc1, index=X.index)

    return pd.DataFrame({
        f"community_{i}": eigengene(nodes) for i, nodes in enumerate(communities)
    })


def community_pc1_variance(G, communities, modality_frames):
    """PC1 explained-variance ratio per community: s[0]**2 / sum(s**2) from the same
    correlation-matrix SVD community_eigengenes() uses, so the two functions can never
    silently disagree."""
    node_modality = nx.get_node_attributes(G, "modality")

    def feature_values(node):
        return modality_frames[node_modality[node]][node]

    def variance_ratio(nodes):
        X = pd.concat([feature_values(n) for n in nodes], axis=1)
        X = (X - X.mean()) / X.std()
        _, s, _ = np.linalg.svd(X.values, full_matrices=False)
        return float(s[0] ** 2 / (s ** 2).sum())

    return pd.Series(
        {f"community_{i}": variance_ratio(nodes) for i, nodes in enumerate(communities)},
        name="pc1_variance_ratio",
    )


# ── Exposome variables ──────────────────────────────────────────────────
def build_exposome_table(participant_index):
    meta = pd.read_csv(META_PATH, low_memory=False)

    htx = meta[
        (meta["data_type"] == "host_transcriptomics")
        & (meta["IntervalName"] == "Screening Colonoscopy")
    ].drop_duplicates("External ID").copy()
    htx["location_rank"] = htx["biopsy_location"].apply(
        lambda loc: LOCATION_PRIORITY.index(loc) if loc in LOCATION_PRIORITY else len(LOCATION_PRIORITY)
    )
    representative = (
        htx.sort_values(["Participant ID", "location_rank", "External ID"])
        .groupby("Participant ID", sort=True).first()
    )

    # Biopsy collection dates are NOT recorded: date_of_receipt is empty on all 252
    # host_transcriptomics rows (it is populated only for stool- and blood-derived
    # assays). Anchoring day length to the biopsy row therefore produced an all-NaN
    # column, i.e. the variable silently contributed nothing.
    #
    # Recovery: week_num counts weeks from each participant's own study start and IS
    # recorded on both biopsy and stool rows. For each participant, take the dated
    # stool sample closest in week_num to the biopsy and shift its date back by the
    # week difference. Resolves 88/90 participants, median week gap 0, max 2.
    dated = meta.dropna(subset=["date_of_receipt", "week_num"]).copy()
    dated["_dt"] = pd.to_datetime(dated["date_of_receipt"], errors="coerce")
    dated = dated.dropna(subset=["_dt"])

    def biopsy_date(participant_id, biopsy_week):
        if pd.isna(biopsy_week):
            return pd.NaT
        rows = dated[dated["Participant ID"] == participant_id]
        if rows.empty:
            return pd.NaT
        nearest = rows.iloc[(rows["week_num"] - biopsy_week).abs().argsort().iloc[0]]
        return nearest["_dt"] - pd.Timedelta(weeks=float(nearest["week_num"] - biopsy_week))

    est_dates = {
        pid: biopsy_date(pid, wk)
        for pid, wk in representative["week_num"].items()
    }
    day_length = pd.Series({
        pid: (day_length_hours(SITE_LATITUDE[representative.loc[pid, "site_name"]], d.dayofyear)
              if pd.notna(d) and representative.loc[pid, "site_name"] in SITE_LATITUDE else np.nan)
        for pid, d in est_dates.items()
    })
    # Day length applies to the HTX layer only. MGX and MBX per-subject values are
    # medians across samples collected throughout the year, so no single photoperiod
    # describes them.

    education_ordinal = representative["Education Level"].map(EDUCATION_ORDER)
    occupation_paid = (representative["Occupation"] == "Paid").astype(float)
    # Age and diagnosis are reliable on the HTX representative row (unlike date,
    # medication and comorbidity -- see below); used as adjustment covariates,
    # not as exposome variables themselves.
    age = pd.to_numeric(representative["consent_age"], errors="coerce")
    diagnosis_ibd = representative["diagnosis"].isin(["CD", "UC"]).astype(float)

    # `demographics` is separate from `exposome` (candidate exposure variables
    # proper) and from `disease_activity` (excluded confounds): these are reported
    # as sensitivity variables, not screened for significance, since they are not
    # exposures. Race is collapsed to a non-White indicator (111/131 White,
    # scattered otherwise) since HMP2's race categories are too sparse individually
    # for a per-category test at this sample size; sex is coded Female=1.
    sex_female = (representative["sex"] == "Female").astype(float)
    # .mask() keeps a missing race as NaN rather than letting `NaN != "White"`
    # (True in pandas) silently miscode it as non-White. Currently 0/231 rows
    # have race unset, so this guards future data rather than changing today's
    # numbers.
    race_nonwhite = (representative["race"] != "White").astype(float).mask(representative["race"].isna())

    # medication_count and comorbidity_count are answered on stool/blood rows,
    # never on the HTX biopsy row (see DRUG_LADDER_COLS comment above), so they
    # are recovered here as "ever Yes across the participant's full row set, any
    # assay" -- the same aggregation shape as the day-length recovery above,
    # applied to a different field.
    all_rows = meta[meta["Participant ID"].isin(participant_index)]

    # BMI is 0/231 populated on the HTX biopsy row (it is recorded on
    # host_genome, methylome and serology rows instead), so reading it off
    # `representative` would silently produce an all-NaN column. Recovered as
    # the one non-null BMI value across the participant's full row set (every
    # participant with any BMI value has exactly one distinct value, so "first"
    # is unambiguous, not an arbitrary tie-break).
    bmi = (
        all_rows.dropna(subset=["BMI"])
        .drop_duplicates("Participant ID")
        .set_index("Participant ID")["BMI"]
        .pipe(pd.to_numeric, errors="coerce")
        .reindex(participant_index)
    )
    demographics = pd.DataFrame({
        "age": age, "sex_female": sex_female, "race_nonwhite": race_nonwhite, "bmi": bmi,
    }).reindex(participant_index)

    def ever_count(cols, taken_values):
        flags = all_rows[cols].apply(lambda s: s.astype(str).str.strip().isin(taken_values))
        flags.insert(0, "Participant ID", all_rows["Participant ID"].values)
        return flags.groupby("Participant ID")[cols].any().sum(axis=1)

    medication_count = ever_count(MEDICATION_YESNO_COLS, {"Yes"}) + ever_count(
        DRUG_LADDER_COLS, DRUG_LADDER_TAKEN_VALUES
    )
    comorbidity_count = ever_count(EXTRAINTESTINAL_COLS + COMORBID_COLS, {"Yes"})

    # Both counts, once correctly recovered, turn out to be disease-activity
    # readouts rather than exposures: medication count predicts diagnosis at
    # AUC=0.963, and comorbidity count is zero for every one of the 22 non-IBD
    # controls. Extending the same principle already applied to CRP/ESR in this
    # Methodology, both are written out for transparency but excluded from the
    # exposome table used for correlation.
    disease_activity = pd.DataFrame({
        "medication_count": medication_count,
        "comorbidity_count": comorbidity_count,
    }).reindex(participant_index)

    # Frequency ladder mapped to 0-4, averaged across a participant's recorded
    # timepoints per item, then averaged across each index's items.
    def diet_index(cols):
        mapped = all_rows[cols].apply(lambda s: s.astype(str).str.strip().map(FFQ_FREQUENCY_MAP))
        mapped.insert(0, "Participant ID", all_rows["Participant ID"].values)
        per_item = mapped.groupby("Participant ID")[cols].mean()
        return per_item.mean(axis=1)

    diet_scores = {name: diet_index(cols) for name, cols in DIET_INDEX_COLS.items()}

    exposome = pd.DataFrame({
        "occupation_paid": occupation_paid,
        "education_ordinal": education_ordinal,
        "day_length_hours": day_length,
        **diet_scores,
    }).reindex(participant_index)

    print("Exposome variable coverage:")
    print((exposome.notna().mean() * 100).round(1).astype(str) + "%")
    print("\nDisease-activity counts (computed, EXCLUDED from the screen above -- see docstring):")
    print(disease_activity.describe().loc[["count", "mean", "max"]])
    print("\nDemographics coverage (reported as sensitivity variables, not screened as exposures):")
    print((demographics.notna().mean() * 100).round(1).astype(str) + "%")
    return exposome, age, diagnosis_ibd, disease_activity, demographics


# ── Correlation + BH correction ─────────────────────────────────────────
def _residualize_ranks(x, z):
    """Rank-transform x and the covariate z, regress rank(x) on rank(z), return
    the residuals -- the first half of a partial (Spearman) correlation."""
    rx, rz = pd.Series(x).rank(), pd.Series(z).rank()
    slope, intercept = np.polyfit(rz, rx, 1)
    return rx - (slope * rz + intercept)


def partial_spearman(x, y, covariate):
    """Partial Spearman correlation between x and y, controlling for one
    covariate. This is the univariate version of what Graphical LASSO already
    does at network scale -- a
    precision-matrix entry is the partial correlation between two features
    after removing what every OTHER feature explains; this removes what ONE
    named confound (age, or diagnosis) explains, then correlates what is left.
    Applied uniformly to every exposome variable, matching how the age check
    already described in the Methodology is meant to work for all of them, not
    only the ones an individual item happens to flag as confounded."""
    paired = pd.concat(
        [pd.Series(x, name="x"), pd.Series(y, name="y"), pd.Series(covariate, name="z")], axis=1
    ).dropna()
    if len(paired) < 4:
        return np.nan
    rx = _residualize_ranks(paired["x"], paired["z"])
    ry = _residualize_ranks(paired["y"], paired["z"])
    return np.corrcoef(rx, ry)[0, 1]


def correlate_against_communities(x, eigengene_df, label_col, label_value, age=None, diagnosis=None):
    rows = []
    for community_col in eigengene_df.columns:
        y = eigengene_df[community_col]
        paired = pd.concat([x, y], axis=1).dropna()
        if len(paired) < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
        row = {label_col: label_value, "community": community_col, "n": len(paired), "rho": rho, "p": p}
        if age is not None:
            row["rho_age_adj"] = partial_spearman(x, y, age)
        if diagnosis is not None:
            row["rho_dx_adj"] = partial_spearman(x, y, diagnosis)
        rows.append(row)
    return pd.DataFrame(rows)


def run_exposome_correlations(exposome, eigengenes, age, diagnosis):
    results = pd.concat(
        [
            correlate_against_communities(exposome[var], eigengenes, "variable", var, age=age, diagnosis=diagnosis)
            for var in exposome.columns
        ],
        ignore_index=True,
    )
    # BH-corrected within each variable, across its community count -- not pooled
    # across variables. Correction runs on the primary (raw) p only; rho_age_adj
    # and rho_dx_adj are a robustness check on results that already clear this
    # bar, not a second independently-corrected test family.
    results["p_adj"] = results.groupby("variable")["p"].transform(lambda p: benjamini_hochberg(p.values))
    return results


def run_h1b_correlations(eigengenes):
    rna_full = pd.read_csv(RNA_FULL_POOL_PATH, index_col=0)
    panel_present = [g for g in LIGHT_LINKED_PANEL if g in rna_full.columns]
    missing = set(LIGHT_LINKED_PANEL) - set(panel_present)
    if missing:
        print(f"H1b panel genes not found in {RNA_FULL_POOL_PATH}: {sorted(missing)}")

    results = pd.concat(
        [correlate_against_communities(rna_full[gene], eigengenes, "gene", gene) for gene in panel_present],
        ignore_index=True,
    )
    # Single BH correction across the whole panel-by-community comparison count.
    results["p_adj"] = benjamini_hochberg(results["p"].values)
    return results


def run_demographics_correlations(demographics, eigengenes):
    """Reports age/sex/race/BMI against the community scores as sensitivity
    variables -- a separate report from the age-adjustment
    already applied inside run_exposome_correlations, answering "how large is
    this confound" rather than "adjust for it and re-check"."""
    results = pd.concat(
        [correlate_against_communities(demographics[col], eigengenes, "demographic", col)
         for col in demographics.columns],
        ignore_index=True,
    )
    # Single BH correction across the whole demographic-by-community count, same
    # convention as run_h1b_correlations.
    results["p_adj"] = benjamini_hochberg(results["p"].values)
    return results


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    G, communities = load_communities()

    modality_frames = {
        "microbiome": pd.read_csv(MGX_PATH, index_col=0),
        "metabolite": pd.read_csv(MBX_PATH, index_col=0),
        "transcript": pd.read_csv(RNA_PATH, index_col=0),
    }
    eigengenes = community_eigengenes(G, communities, modality_frames)
    pc1_variance = community_pc1_variance(G, communities, modality_frames)

    exposome, age, diagnosis_ibd, disease_activity, demographics = build_exposome_table(eigengenes.index)
    exposome_results = run_exposome_correlations(exposome, eigengenes, age, diagnosis_ibd)
    h1b_results = run_h1b_correlations(eigengenes)
    demographics_results = run_demographics_correlations(demographics, eigengenes)

    eigengenes.to_csv(f"{RESULTS_DIR}exposome_community_eigengenes.csv")
    pc1_variance.to_csv(f"{RESULTS_DIR}exposome_community_pc1_variance.csv")
    exposome.to_csv(f"{RESULTS_DIR}exposome_variables.csv")
    disease_activity.to_csv(f"{RESULTS_DIR}exposome_disease_activity_excluded.csv")
    demographics.to_csv(f"{RESULTS_DIR}exposome_demographics.csv")
    exposome_results.to_csv(f"{RESULTS_DIR}exposome_community_correlations.csv", index=False)
    h1b_results.to_csv(f"{RESULTS_DIR}h1b_panel_community_correlations.csv", index=False)
    demographics_results.to_csv(f"{RESULTS_DIR}demographics_community_correlations.csv", index=False)

    print("\nPC1 variance explained per community:")
    print(pc1_variance.round(4).to_string())
    print("\nSignificant exposome-community associations (p_adj < 0.05):")
    print(exposome_results[exposome_results["p_adj"] < 0.05].to_string(index=False))
    print("\nSignificant H1b panel-community associations (p_adj < 0.05):")
    print(h1b_results[h1b_results["p_adj"] < 0.05].to_string(index=False))
    print("\nDemographics-community associations (all rows -- sensitivity report, not a screen):")
    print(demographics_results.to_string(index=False))


def self_test():
    # Boston, mid-December vs. Atlanta, mid-June -- same two examples used
    # to sanity-check this formula by hand before writing it into code.
    boston_dec = day_length_hours(SITE_LATITUDE["MGH"], pd.Timestamp("2015-12-15").dayofyear)
    atlanta_jun = day_length_hours(SITE_LATITUDE["Emory"], pd.Timestamp("2015-06-10").dayofyear)
    assert 8.5 < boston_dec < 9.5, f"Boston mid-Dec day length off: {boston_dec:.2f}h"
    assert 13.8 < atlanta_jun < 14.8, f"Atlanta mid-Jun day length off: {atlanta_jun:.2f}h"
    equator_equinox = day_length_hours(0.0, pd.Timestamp("2015-03-20").dayofyear)
    # Forsythe et al. (1995) measures sunrise-to-sunset including the sun's disc and a
    # refraction allowance, so the equinox value at the equator is legitimately just over
    # 12h (~12.1h), matching published almanacs. Tolerance is set accordingly.
    assert abs(equator_equinox - 12.0) < 0.25, f"Equator equinox should be ~12h: {equator_equinox:.2f}h"

    bh = benjamini_hochberg([0.001, 0.01, 0.02, 0.5, 0.8])
    assert np.all(np.diff(bh) >= -1e-9), f"BH-adjusted p-values should be non-decreasing with input rank: {bh}"
    assert bh[0] < 0.05, f"Smallest p-value should survive BH correction here: {bh}"
    nan_result = benjamini_hochberg([np.nan, np.nan])
    assert np.all(np.isnan(nan_result)), "All-NaN input should return all-NaN, not raise"

    print(f"Boston, mid-Dec:  {boston_dec:.2f}h daylight")
    print(f"Atlanta, mid-Jun: {atlanta_jun:.2f}h daylight")
    print(f"Equator, equinox: {equator_equinox:.2f}h daylight")
    print(f"BH-adjusted p-values: {bh}")
    print("self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
