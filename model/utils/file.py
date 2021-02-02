import os
import csv


def resolve_path(path):
    """Creates folders for given path, if they don't exists.

    Args:
        path (path): Path to be created.
    """

    full_path = ""
    for name in path.split("/"):
        if "." not in name:
            full_path += name + "/"
            if not os.path.exists(full_path):
                os.mkdir(full_path)


def save_results_to_csv(file_path, results, headers):
    """Writes results to a csv file.

    Args:
        file_path (str): File destination path.
        results (list of list): Matrix with results to be writen.
        headers (list of str): Header names.
    """

    resolve_path(file_path)

    with open(file_path, "w") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(headers)

        for row in results:
            csv_writer.writerow(row)
