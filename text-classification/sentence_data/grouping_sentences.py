import os
import csv

sentiments = {'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
all_sentences = []

for subdir in os.listdir("all"):
    subpath = os.path.join("all", subdir)
    if os.path.isdir(subpath):
        for file in os.listdir(subpath):
            if file != ".DS_Store":
                file_path = os.path.join(subpath, file)
                with open(file_path, 'r') as f:
                    sentence = f.read().strip()
                label = sentiments[subdir]
                all_sentences.append([sentence, label])

with open('SA_data.csv', mode='w') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for sentence, label in all_sentences:
        writer.writerow([sentence, label])
