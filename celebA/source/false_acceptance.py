import numpy as np
import os
import argparse


if __name__ == '__main__':

    '''
        Make sure to compute cosine similarity scores and saving them as numpy arrays in r_at_k.py first prior to 
        computing false acceptance scores in this script.
    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DF-Net")
    parser.add_argument("--feature", type=str, default="eyes")
    parser.add_argument("--label_file", type=str, default="identity_CelebA.txt", help="file detailing identity of each image")
    parser.add_argument("--write_file", type=str, default='false_acceptance.txt', help="where to write false acceptance scores to.")
    parser.add_argument("--test_image_dir", type=str, default="celebA/images/inpainted/face/EdgeConnect/")
    parser.add_argument("--db_image_dir", type=str, default="celebA/images/original/")

    args = parser.parse_args()

    db_image_names = list(os.listdir(args.db_image_dir))
    db_image_names.sort()
    test_image_names = list(os.listdir(args.test_image_dir))
    test_image_names.sort()

    d = {}
    with open(args.label_file) as fp:
        line = fp.readline().strip()
        while line:
            k, v = line.split(' ')
            d[k] = v
            line = fp.readline().strip()

    if args.model == 'original':
        similarities = 'original_original_similarities.npy'
    else:
        similarities = np.load('{}_{}_original_similarities.npy'.format(args.model, args.feature))
    total_correct = []
    for i in range(len(similarities)):
        query_name = test_image_names[i]
        row = similarities[i]
        closest_2 = row.argsort()[-2:][::-1]
        retrieved_name = db_image_names[closest_2[0]]
        retrieved_score = row[closest_2[0]]
        if retrieved_name == query_name:
            retrieved_name = db_image_names[closest_2[1]]
            retrieved_score = row[closest_2[1]]
        if d[query_name] == d[retrieved_name]:
            correct = 1
        else:
            correct = 0
        total_correct.append((retrieved_score, correct))

    total_correct.sort()
    total_correct.reverse()
    fa_1, fa_10 = len(total_correct) // 100, len(total_correct) // 10
    ta_counter = fa_counter = 0
    fa_1_measured = fa_10_measured = False
    for i in range(len(total_correct)):
        correct = total_correct[i][1]
        if correct == 0:
            fa_counter += 1
        else:
            ta_counter += 1
        if fa_counter == fa_1 and not fa_1_measured:
            with open(args.write_file, 'a+') as f:
                f.write('{} {} FA 1%: {}\n'.format(args.model, args.feature, ta_counter / i))
            fa_1_measured = True
        if fa_counter == fa_10 and not fa_10_measured:
            with open(args.write_file, 'a+') as f:
                f.write('{} {} FA 10%: {}\n'.format(args.model, args.feature, ta_counter / i))
            fa_10_measured = True

