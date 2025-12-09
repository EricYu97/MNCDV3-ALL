"""
For different log files:
"""
import re
import pandas as pd
import glob

pattern = r"Model_TestedAccuracy=([\d\.]+), Precision=([\d\.]+), Recall=([\d\.]+), cF1=([\d\.]+), ciou=([\d\.]+)"

rows = []

for log_file in glob.glob("*.log"):
    model_name = log_file.replace(".log", "")
    with open(log_file, "r") as f:
        text = f.read()

    match = re.search(pattern, text)
    if match:
        acc, prec, rec, cf1, ciou = match.groups()
        rows.append({
            "Model": model_name,
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "cF1": float(cf1),
            "cIoU": float(ciou)
        })

df = pd.DataFrame(rows)
df.to_csv("results.csv", index=False)
print("Done! Saved to results.csv")


"""
For all logs in one File:
"""
import re
import pandas as pd

# Your combined log file
LOG_FILE = "all_models.log"   # <-- change this if needed

pattern = r"Model_Tested=([\w\-]+), Accuracy=([\d\.]+), Precision=([\d\.]+), Recall=([\d\.]+), cF1=([\d\.]+), ciou=([\d\.]+)"

rows = []

with open(LOG_FILE, "r") as f:
    text = f.read()

matches = re.findall(pattern, text)

for m in matches:
    model, acc, prec, rec, cf1, ciou = m
    rows.append({
        "Model": model,
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "cF1": float(cf1),
        "cIoU": float(ciou),
    })

df = pd.DataFrame(rows)
df.to_csv("results.csv", index=False)

print(f"Extracted {len(rows)} entries → saved to results.csv")