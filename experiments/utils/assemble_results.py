import argparse
import csv
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "description_file_path", help="File describing the single-run results. \
                                   Every row in this file should provide \
                                   the path for an individual result file.")

parser.add_argument(
    "experiment", help="Experiment (a.k.a training strategy) used.")

args = parser.parse_args()

description_file_path = args.description_file_path
experiment = args.experiment
summary_file_path = "results/" + experiment + "/summary.csv"

summary = {}

# Load every result for later calculation
with open(description_file_path, "r") as description_file:
    for row in description_file:
        result_file_path = row.split()[2]

        with open(result_file_path, "r") as result_file:
            csv_reader = csv.reader(result_file)
            # Discard header
            next(csv_reader)

            for row in csv_reader:
                subject_id, loss, acc = row

                if subject_id not in summary:
                    summary[subject_id] = []

                summary[subject_id].append([loss, acc])

# arr = np.array(summary["900"], dtype=np.float)
# print(np.mean(arr, axis=0))
# exit()

# Compute means and stds along with all results
with open(summary_file_path, "w") as summary_file:
    csv_writer = csv.writer(summary_file)
    csv_writer.writerow([
        "subject_id",
        "loss_mean",
        "acc_mean",
        "loss_std",
        "acc_std",
    ])

    for subject_id, _values in summary.items():
        values = np.array(_values, dtype=np.float)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)

        csv_writer.writerow([subject_id, *mean, *std])
