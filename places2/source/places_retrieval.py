import argparse
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_features", type=str,
                        default=r"places2\feature_vectors\standard\places2_0.2_DF-Net.dat",
                        help='path to saved numpy array of the query features')
    parser.add_argument("--database_features", type=str, default="places2/feature_vectors/standard/db_places_encodings.dat",
                        help='path to saved numpy array of the database features')
    parser.add_argument("--query_labels", type=str, default="places2/classification_labels.dat",
                        help='path to a list of the query labels (pickle file)')
    parser.add_argument("--database_filenames", type=str, default="places2/db_filenames.dat",
                        help='path to a list of the database filenames (pickle file)')
    parser.add_argument("--places2_classes_file", type=str, default="places2/class_list.txt")

    args = parser.parse_args()

    # Extract the database labels from their file paths
    label_dict = {}
    with open(args.places2_classes_file, 'r') as f:
        for line in f:
            key, val = line.split()
            label_dict[key[3:]] = val
    with open(args.database_filenames, 'rb') as f:
        db_filenames = pickle.load(f)
    db_labels = []
    for filename in db_filenames:
        split = filename.split('/')
        if len(split) == 4:
            db_label = split[2]
        else:
            db_label = split[2] + '/' + split[3]
        db_labels.append(label_dict[db_label])

    # Load query labels
    with open(args.query_labels, 'rb') as f:
        query_labels = pickle.load(f)

    # Load database features
    with open(args.database_features, 'rb') as f:
        db_encodings = pickle.load(f)
        db_encodings = normalize(db_encodings)
        assert len(db_encodings) == len(db_labels), \
            'Number of database features must equal number of corresponding labels'

    # Load query features
    with open(args.query_features, 'rb') as f:
        query_features = np.squeeze(pickle.load(f))
        query_features = normalize(query_features)
        assert len(query_features) == len(query_labels), \
            'Number of query features must equal number of corresponding labels'

    similarities = cosine_similarity(query_features, db_encodings)
    aps = []
    for i in range(len(query_features)):
        query_label = query_labels[i]
        count = db_labels.count(query_label)
        top_k = similarities[i].argsort()[-count:][::-1]
        neighbor_labels = np.array(db_labels)[top_k]
        correct = (neighbor_labels == query_label).astype(int)
        k_count = np.cumsum(correct, 0)
        ap = np.sum(k_count / np.arange(1, count + 1) * correct) / count
        aps.append(ap)
    mAP = np.mean(aps)
    print('Mean Average Precision: {}'.format(mAP))

