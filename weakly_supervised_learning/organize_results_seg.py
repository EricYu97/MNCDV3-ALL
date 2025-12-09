import re
import pandas as pd
import csv
# Combined log file
LOG_PATH = "./output_logs/test/Testing-MNCDV3-Seg-15486576.out"

results = []

with open(LOG_PATH) as f:
    for line in f:
        line = line.strip()
        if line.startswith("Model_Tested:"):
            # Extract model
            model_match = re.search(r"Model_Tested:\s*([^,]+),", line)
            model = model_match.group(1).strip() if model_match else "Unknown"

            # Extract per-class F1
            f1_match = re.search(r"tensor\(\[([0-9\.,\s]+)\]", line)
            per_class_f1 = [float(x) for x in f1_match.group(1).split(",")] if f1_match else []

            # Extract average F1
            avg_match = re.search(r"Average F1 Score[^0-9]*([0-9\.]+)", line)
            avg_f1 = float(avg_match.group(1)) if avg_match else None

            results.append({
                "model": model,
                "per_class_f1": per_class_f1,
                "average_f1": avg_f1
            })

# Step 2: Write to CSV
if results:
    num_classes = max(len(r["per_class_f1"]) for r in results)
    header = ["model"] + [f"f1_class_{i+1}" for i in range(num_classes)] + ["average_f1"]

    with open("f1_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            # Pad per-class F1 if some entries have fewer classes
            row = [r["model"]] + r["per_class_f1"] + ["" for _ in range(num_classes - len(r["per_class_f1"]))] + [r["average_f1"]]
            writer.writerow(row)

print("CSV file 'f1_results.csv' has been created!")
