"""Build the canonical, deterministic map of the Zhong et al. 2025 dataset.

Single source of truth the whole team (and the graph port / no-code explorer)
can read: every recording -> cohort / stage / mouse, every brain-area code ->
region, every stimulus role -> meaning. Nothing here is hand-assigned; it is
derived from the release inventory + the reproduction code, with provenance.

Sources (authoritative):
  - zhong2025/assets/imaging-experiment-index.json  (89 recordings, experiment labels)
  - original/utils.py:312  neu_area_ID           -> brain-area code grouping
  - original/utils.py:419/505 + get_cat_id:137    -> stimulus role (stim_id) convention

Run:  python scripts/build_canonical_map.py
Emits: zhong2025/assets/canonical_map.json   (machine-readable, for code/explorer)
       references/canonical_map.md            (human-readable, for the team to verify)
"""
from __future__ import annotations
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(HERE, "zhong2025", "assets", "imaging-experiment-index.json")
OUT_JSON = os.path.join(HERE, "zhong2025", "assets", "canonical_map.json")
OUT_MD = os.path.join(HERE, "references", "canonical_map.md")

# --- authoritative brain-area map (original/utils.py:312 neu_area_ID) -------------
# Group-level only: individual ints are NOT mapped to PM/AM/MMA anywhere in the
# released code, so we do not claim per-int anatomy.
AREAS = {
    "V1":  {"iarea": [8],          "region": "V1 (primary visual cortex)",
            "role": "detects visual input & novelty; prespecified primary area"},
    "mHV": {"iarea": [0, 1, 2, 9], "region": "medial higher visual areas (PM / AM / MMA, + lateral RSC collectively)",
            "role": "greatest learning-related plasticity / discrimination (the effect localizes here)"},
    "lHV": {"iarea": [5, 6],       "region": "lateral higher visual areas",
            "role": "strong novelty response"},
    "aHV": {"iarea": [3, 4],       "region": "anterior higher visual areas",
            "role": "reward-prediction / task signals (supervised mice only)"},
    "unassigned": {"iarea": [-1, 7], "region": "outside any retinotopically mapped area",
            "role": "excluded from area analyses (-1 = no area)"},
}
# A wrong map the team used in early exploration -- kept here so the correction is explicit.
KNOWN_WRONG_AREA_MAP = {"1": "V1", "2": "mHV", "3": "aHV",
                        "note": "INCORRECT — do not use. Real codes are in AREAS above (V1=8, not 1)."}

# --- authoritative stimulus-role map (original/utils.py:419,505; get_cat_id:137) --
# Roles are FUNCTIONAL and assigned per session by which wall is rewarded
# (get_cat_id maps the rewarded wall -> leaf1). They are NOT physical textures.
# NOTE on 'rewarded': it marks the REFERENCE role only. leaf2/leaf3/leaf1_swap are
# leaf-FAMILY test stimuli introduced WITHOUT reward (generalization tests), and whether
# a wall pays reward is set per session by get_cat_id — so 'rewarded' is a role-family
# label, NOT per-session reward truth. See `family` and each `meaning`.
ROLES = {
    0: {"name": "circle1", "family": "circle", "meaning": "unrewarded reference texture", "rewarded": False},
    1: {"name": "circle2", "family": "circle", "meaning": "circle-family test texture (generalization) — unrewarded", "rewarded": False},
    2: {"name": "leaf1",   "family": "leaf",   "meaning": "the REWARDED reference texture (licking here delivers water)", "rewarded": True},
    3: {"name": "leaf2",   "family": "leaf",   "meaning": "leaf-family test texture — INTRODUCED WITHOUT reward (Test 1 generalization)", "rewarded": False},
    4: {"name": "leaf3",   "family": "leaf",   "meaning": "further leaf-family test texture (Test 2) — introduced without reward", "rewarded": False},
    5: {"name": "leaf1_swap1", "family": "leaf", "meaning": "swapped/modified leaf1 control texture", "rewarded": False},
    6: {"name": "leaf1_swap2", "family": "leaf", "meaning": "swapped/modified leaf1 control texture", "rewarded": False},
}
DPRIME_PAIR = {"rewarded": 2, "unrewarded": 0,
               "note": "selectivity d' compares leaf1 (role 2) vs circle1 (role 0); stim_ID=[2,0] in utils.py. "
                       "'rewarded' marks the reference roles only — leaf-family test stimuli are introduced unrewarded."}

COHORTS = {
    "sup":     "Supervised / task: water-deprived, licking in the leaf corridor delivers water (reward-driven).",
    "unsup":   "Unsupervised: same stimuli, NOT water-deprived, no reward (exposure only).",
    "naive":   "Naive baseline: imaged before/without the training protocol (OVERLAPS other cohorts — a naive recording reuses a sup/unsup/grating mouse).",
    "grating": "Grating control: a separate control group shown grating stimuli.",
}
COHORT_PRIORITY = ["sup", "unsup", "grating", "naive"]  # training cohort wins over naive-baseline role


def cohort_family(label: str) -> str:
    if label.startswith("sup_"):
        return "sup"
    if label.startswith("unsup_"):
        return "unsup"
    if label.startswith("naive_"):
        return "naive"
    if "grating" in label:
        return "grating"
    return "unknown"


def parse_stage(label: str):
    """Return (phase, moment) e.g. ('Train 1', 'before'), ('Test 1', ''). Deterministic."""
    s = label
    for pre in ("sup_", "unsup_", "naive_"):
        if s.startswith(pre):
            s = s[len(pre):]
    s = s.replace("_grating", "")
    s = s.replace("_learning", "")
    moment = ""
    if s.endswith("_before"):
        moment, s = "before", s[:-len("_before")]
    elif s.endswith("_after"):
        moment, s = "after", s[:-len("_after")]
    phase_map = {
        "train1": "Train 1", "train2": "Train 2",
        "test1": "Test 1", "test2": "Test 2", "test3": "Test 3",
    }
    phase = phase_map.get(s, s)
    return phase, moment


def friendly(recording_id, mouse, cohort, phase, moment):
    coh = {"sup": "Supervised", "unsup": "Unsupervised", "naive": "Naive", "grating": "Grating"}[cohort]
    stage = phase + ((" · " + moment + "-learning") if moment else "")
    return f"{coh} mouse {mouse} — {stage}"


def main():
    idx = json.load(open(INDEX))
    exps = idx["experiments"]

    # recording_id -> record (aggregated across every label it appears under)
    recs = {}
    for label, entries in exps.items():
        fam = cohort_family(label)
        phase, moment = parse_stage(label)
        for e in entries:
            s = e["source"]
            rid = e["recording_id"]
            r = recs.setdefault(rid, {
                "recording_id": rid,
                "mouse": s["mname"],
                "date": s["datexp"],
                "block": str(s.get("blk")),
                "sess_num": s.get("sess#"),
                "gender": s.get("Gender"),
                "depth_um": s.get("depth"),
                "reward_type": s.get("rewType"),
                "stim_id": s.get("stim_id"),
                "retinotopy_id": e.get("retinotopy_id"),
                "note": s.get("Note"),
                "memberships": [],   # list of {cohort, phase, moment, label}
            })
            r["memberships"].append({"cohort": fam, "phase": phase, "moment": moment, "label": label})

    # derive primary cohort per recording + friendly label
    for rid, r in recs.items():
        fams = {m["cohort"] for m in r["memberships"]}
        primary = next((c for c in COHORT_PRIORITY if c in fams), sorted(fams)[0])
        r["cohort"] = primary
        r["also"] = sorted(fams - {primary})
        # pick the membership from the primary cohort for the friendly stage label
        pm = next((m for m in r["memberships"] if m["cohort"] == primary), r["memberships"][0])
        r["stage"] = (pm["phase"] + ((" · " + pm["moment"]) if pm["moment"] else "")).strip()
        r["friendly"] = friendly(rid, r["mouse"], primary, pm["phase"], pm["moment"])

    # mouse -> rollup
    mice = {}
    for r in recs.values():
        m = mice.setdefault(r["mouse"], {"mouse": r["mouse"], "gender": r["gender"],
                                          "cohorts": set(), "recordings": []})
        m["cohorts"].add(r["cohort"])
        for mem in r["memberships"]:
            m["cohorts"].add(mem["cohort"])
        m["recordings"].append(r["recording_id"])
    for m in mice.values():
        fams = m["cohorts"]
        m["primary_cohort"] = next((c for c in COHORT_PRIORITY if c in fams), sorted(fams)[0])
        m["also"] = sorted(fams - {m["primary_cohort"]})
        m["cohorts"] = sorted(fams)
        m["n_recordings"] = len(set(m["recordings"]))
        m["recordings"] = sorted(set(m["recordings"]))

    out = {
        "_meta": {
            "title": "Zhong et al. 2025 — canonical deterministic map",
            "generated_by": "scripts/build_canonical_map.py",
            "sources": {
                "recordings": "zhong2025/assets/imaging-experiment-index.json",
                "areas": "original/utils.py:312 neu_area_ID",
                "roles": "original/utils.py:419,505 + get_cat_id:137",
            },
            "counts": {"recordings": len(recs), "mice": len(mice)},
            "warning": "Cohorts overlap: a 'naive' recording reuses a sup/unsup/grating mouse. See mice[].also.",
        },
        "cohorts": COHORTS,
        "brain_areas": AREAS,
        "brain_areas_known_wrong": KNOWN_WRONG_AREA_MAP,
        "stimulus_roles": {str(k): v for k, v in ROLES.items()},
        "dprime_pair": DPRIME_PAIR,
        "mice": {k: mice[k] for k in sorted(mice)},
        "recordings": {k: recs[k] for k in sorted(recs)},
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    # ---- human-readable markdown the team can eyeball ----
    L = []
    L.append("# Zhong et al. 2025 — canonical map (auto-generated, do not hand-edit)\n")
    L.append(f"_Generated by `scripts/build_canonical_map.py` from the release inventory + reproduction code._\n")
    L.append(f"**{len(recs)} recordings · {len(mice)} mice.** Cohorts overlap — see the ‘also’ column.\n")

    L.append("\n## Cohorts\n")
    for k, v in COHORTS.items():
        L.append(f"- **{k}** — {v}")

    L.append("\n## Brain-area codes (`iarea` → region)  ⚠️ authoritative\n")
    L.append("| group | iarea codes | region | role |")
    L.append("|---|---|---|---|")
    for k, v in AREAS.items():
        L.append(f"| `{k}` | {v['iarea']} | {v['region']} | {v['role']} |")
    L.append(f"\n> ⚠️ The early team map `{{1:'V1',2:'mHV',3:'aHV'}}` is **wrong** — V1 is code **8**, not 1. Use the table above.\n")

    L.append("\n## Stimulus roles (`stim_id` → meaning)  — functional, assigned per session\n")
    L.append("| stim_id | role | meaning | rewarded |")
    L.append("|---|---|---|---|")
    for k in sorted(ROLES):
        v = ROLES[k]
        L.append(f"| {k} | `{v['name']}` | {v['meaning']} | {'✅' if v['rewarded'] else '—'} |")
    L.append(f"\n> Selectivity **d′ compares `leaf1` (2) vs `circle1` (0)**. Roles come from which wall is rewarded (`get_cat_id`), not the physical texture.\n")

    L.append("\n## Mice (19)\n")
    L.append("| mouse | cohort | also | sex | recordings |")
    L.append("|---|---|---|---|---|")
    for k in sorted(mice):
        m = mice[k]
        L.append(f"| `{m['mouse']}` | **{m['primary_cohort']}** | {', '.join(m['also']) or '—'} | {m['gender'] or '?'} | {m['n_recordings']} |")

    L.append("\n## Recordings (89)\n")
    L.append("| recording_id | friendly name | cohort | stage | also | sess# |")
    L.append("|---|---|---|---|---|---|")
    def sortkey(rid):
        r = recs[rid]
        return (COHORT_PRIORITY.index(r["cohort"]) if r["cohort"] in COHORT_PRIORITY else 9, r["mouse"], r["date"])
    for rid in sorted(recs, key=sortkey):
        r = recs[rid]
        L.append(f"| `{rid}` | {r['friendly']} | {r['cohort']} | {r['stage']} | {', '.join(r['also']) or '—'} | {r['sess_num']} |")

    open(OUT_MD, "w").write("\n".join(L) + "\n")

    # ---- verification spot-checks ----
    print(f"WROTE {OUT_JSON}")
    print(f"WROTE {OUT_MD}")
    print(f"recordings={len(recs)} mice={len(mice)}")
    print("cohort counts (primary):", {c: sum(1 for r in recs.values() if r['cohort'] == c) for c in COHORT_PRIORITY})
    print("mice per primary cohort:", {c: sum(1 for m in mice.values() if m['primary_cohort'] == c) for c in COHORT_PRIORITY})
    for chk in ["VR2", "TX105", "DR10", "TX108", "LZ13"]:
        if chk in mice:
            m = mice[chk]
            print(f"  {chk}: primary={m['primary_cohort']} also={m['also']} n_rec={m['n_recordings']}")


if __name__ == "__main__":
    main()
