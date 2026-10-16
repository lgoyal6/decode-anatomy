from dataclasses import asdict
import re
from serving.score import score_response

def run(x):
    if set(x) != {"text","gold","finish_reason"}: raise ValueError("Invalid input fields")
    if not isinstance(x["text"],str) or len(x["text"])>4096 or not isinstance(x["gold"],str) or not re.fullmatch(r"-?[0-9]{1,32}(?:\.[0-9]{1,16})?",x["gold"]): raise ValueError("Use bounded text and an ungrouped numeric reference")
    if x["finish_reason"] not in ("stop","length","cancelled","error"): raise ValueError("Unknown finish reason")
    return asdict(score_response(x["text"],x["gold"],finish_reason=x["finish_reason"]))
