import csv


def extract_sentences():
    base_path = 'data/dataset.csv'
    save_path = 'data/dataset_sentences'
    all_text = []
    with open(base_path, 'r') as f:
        spam_reader = csv.reader(f, delimiter=',', quotechar='|')
        for row in spam_reader:
            all_text.append(row[2] + row[3])
    with open(save_path, 'a') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(all_text)

