"""Copy the hand grades from results/grading_sheet.csv into results/results.csv."""
import csv

from config import RESULTS_DIR


def main():
    with open(RESULTS_DIR / "grading_key.csv") as f:
        key = {row["sheet_id"]: (row["config"], row["question_id"]) for row in csv.DictReader(f)}
    with open(RESULTS_DIR / "grading_sheet.csv") as f:
        grades = {key[row["sheet_id"]]: row["accuracy"].strip().upper() for row in csv.DictReader(f)}

    with open(RESULTS_DIR / "results.csv") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        runs = list(reader)

    for run in runs:
        grade = grades[(run["config"], run["question_id"])]
        if grade not in ("P", "F"):
            raise ValueError(f"{run['config']} question {run['question_id']} needs a P or F, found {grade!r}")
        run["accuracy"] = grade

    with open(RESULTS_DIR / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runs)
    print(f"Graded {len(runs)} runs")


if __name__ == "__main__":
    main()
