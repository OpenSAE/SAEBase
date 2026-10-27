"""Read M2 provenance-verified raw measurement rows without re-running forwards."""
from __future__ import annotations
from collections import defaultdict
from saebase.interventions import read_results

def calculate_interventions(path, identity, *, conditions):
    rows = read_results(path, identity)
    requested = {c.name for c in conditions}
    records = defaultdict(list)
    refs = defaultdict(list)
    for row in rows:
        name = row["condition_name"]
        if name not in requested:
            continue
        for key,value in row["metrics"].items():
            if isinstance(value, (float,int)) and not isinstance(value,bool):
                records[(name,key)].append(float(value))
                refs[(name,key)].append(row["measurement_id"])
    names = ("logit_delta_l2","logit_delta_max_abs","kl_baseline_to_edited",
             "activation_delta_l2","target_logit_delta","edited_target_top1")
    out = {}
    for condition in conditions:
        for name in names:
            vals = records.get((condition.name,name),[])
            if vals:
                out[(condition.name,"intervention."+name)] = (sum(vals)/len(vals),len(vals),tuple(refs[(condition.name,name)]))
    return out, rows
