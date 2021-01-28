import csv


def save_results_to_csv(file_path, results, headers):
    """Writes results to a csv file.

    Args:
        file_path (str): File destination path.
        results (list of list): Matrix with results to be writen.
        headers (list of str): Header names.
    """

    with open(file_path, "w") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(headers)

        for row in resuts:
            csv_writer.writerow(row)
