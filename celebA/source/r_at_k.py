import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
import argparse


def compute_r_at_k(retrived_list, query_name, k, names, label_dict):
    query_label = label_dict[query_name]
    retrieved_itself = False
    correct = 0
    j = 0
    while (j < k and not retrieved_itself) or (j <= k and retrieved_itself):
        retrieved_name = names[retrived_list[j]]
        if retrieved_name != query_name:
            retrieved_label = label_dict[retrieved_name]
            if retrieved_label == query_label:
                correct = 1
                break
        else:
            retrieved_itself = True
        j += 1
    return correct


if __name__ == '__main__':

    """
        Run encode.py to first save image feature vector that are computed using VGGFace prior to running this script. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--db_encodings", type=str, default="original.npy", help='numpy array of the original '
                                                                                 'image feature vectors')
    parser.add_argument("--model", type=str, default="DF-Net")
    parser.add_argument("--feature", type=str, default="eyes")
    parser.add_argument("--label_file", type=str, default="identity_CelebA.txt",
                        help="file detailing identity of each image")
    parser.add_argument("--inpainted_image_dir", type=str, default="celebA/images/inpainted")
    parser.add_argument("--db_image_dir", type=str, default="celebA/images/original")

    args = parser.parse_args()
    if args.model == 'original':
        test_image_names = list(os.listdir(args.db_image_dir))
    else:
        test_image_names = list(os.listdir(os.path.join(args.inpainted_image_dir, args.feature, args.model)))

    db_image_names = list(os.listdir(args.db_image_dir))
    test_image_names.sort()
    db_image_names.sort()

    d = {}
    with open(args.label_file) as fp:
        line = fp.readline().strip()
        while line:
            k, v = line.split(' ')
            d[k] = v
            line = fp.readline().strip()

    if args.model == 'original':
        embeddings_name = 'original'
    else:
        embeddings_name = '{}_{}'.format(args.model, args.feature)
    cosine_similarities_file = '{}_original_similarities.npy'.format(embeddings_name)

    if os.path.exists(cosine_similarities_file + '.npy'):
        cosine_similarities = np.load(cosine_similarities_file)
    else:
        # generate cosine similarities if don't already have them
        x = np.load('{}.npy'.format(embeddings_name))
        y = np.load(args.db_encodings)
        preprocessed = normalize(np.concatenate([x, y]))
        x = preprocessed[:len(x), :]
        y = preprocessed[len(x):, :]
        cosine_similarities = cosine_similarity(x, y)
        np.save(cosine_similarities_file, cosine_similarities)

    k_values = [1, 10, 100]
    retrieval_scores = [[], [], []]
    assert(len(test_image_names) == len(cosine_similarities))
    for i in range(len(cosine_similarities)):
        model_row = cosine_similarities[i]
        model_ranking = model_row.argsort()[::-1]
        for p in range(len(retrieval_scores)):
            retrieval_scores[p].append(compute_r_at_k(model_ranking, test_image_names[i], k_values[p], db_image_names, d))

    with open(args.write_file, 'a+') as f:
        f.write('{} {} k@1: {}, k@10: {}, k@100: {}\n'.format(args.model, args.feature, retrieval_scores[0],
                                                              retrieval_scores[1], retrieval_scores[2]))
