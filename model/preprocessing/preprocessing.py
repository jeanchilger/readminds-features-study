from sklearn.preprocessing import normalize, LabelEncoder


def normalize_and_encode(dataset, feature_headers, label_header, norm="l2"):
    """[summary]

    Args:
        dataset (pandas.Dataframe): Input dataset.
        feature_headers (list of str): Column names for features.
        label_header (list of str or str): Column name for label.
        norm (str, optional): Norm to use in normalization. Defaults to "l2".

    Returns:
        pandas.Dataframe: Normalized and label-encoded dataset.
    """

    label_encoder = LabelEncoder()
    label_encoder.fit(dataset[label_header])

    dataset[label_header] = label_encoder.transform(dataset[label_header])

    for header in feature_headers:
        dataset[header] = normalize([dataset[header]], norm=norm)[0]

    return dataset