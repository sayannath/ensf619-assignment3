"""Write a shuffled grading sheet that hides which configuration produced each answer."""
import csv
import random

from config import QUESTIONS_FILE, RESULTS_DIR


def main():
    with open(QUESTIONS_FILE) as f:
        expected = {q["id"]: q["expected_answer"] for q in csv.DictReader(f)}
    with open(RESULTS_DIR / "results.csv") as f:
        runs = list(csv.DictReader(f))

    random.Random(619).shuffle(runs)

    with open(RESULTS_DIR / "grading_sheet.csv", "w", newline="") as sheet, \
            open(RESULTS_DIR / "grading_key.csv", "w", newline="") as key:
        sheet_writer = csv.writer(sheet)
        key_writer = csv.writer(key)
        sheet_writer.writerow(["sheet_id", "question", "expected_answer", "answer", "accuracy"])
        key_writer.writerow(["sheet_id", "config", "question_id"])
        for sheet_id, run in enumerate(runs, 1):
            sheet_writer.writerow([sheet_id, run["question"], expected[run["question_id"]], run["answer"], ""])
            key_writer.writerow([sheet_id, run["config"], run["question_id"]])

    print("Fill the accuracy column of results/grading_sheet.csv with P or F, then run apply_grades.py")


if __name__ == "__main__":
    main()
