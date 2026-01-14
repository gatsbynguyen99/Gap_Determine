import json
import os
from typing import Dict, List
import pandas as pd

#Create the frame by frame CSV
#JSON

#converts list of dicts -> pandas DF
#writes results
def write_results_csv(results: List[Dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    path = os.path.join(out_dir, "results.csv")
    df.to_csv(path, index=False)
    return path


#compute total/ accepted/ rejected
#acceptance rate
#stats for accepted frames
def write_summary_json(results: List[Dict], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    total = len(results)
    accepted = sum(1 for r in results if r.get("ok") is True)
    rejected = total - accepted
    acc_rate = (accepted / total) if total else 0.0

    accepted_rows = [r for r in results if r.get("ok") is True and r.get("gap_px") is not None]
    def _stats(key: str):
        vals = [r[key] for r in accepted_rows if key in r and r[key] is not None]
        if not vals:
            return None
        s = pd.Series(vals)
        return {
            "count": int(s.count()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    summary = {
        "total_frames": total,
        "accepted_frames": accepted,
        "rejected_frames": rejected,
        "acceptance_rate": acc_rate,
        "gap_px_stats": _stats("gap_px"),
        "gap_mm_stats": _stats("gap_mm"),
    }

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path
