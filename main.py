import string

import matplotlib.pyplot
import numpy
import pandas

dataset = pandas.read_csv("hns_2018_2019.csv")
data = dataset.iloc[:, :].values
model_data = dataset.iloc[:, [2, 3, 5]].values

training_features = []
testing_features = []
training = []

training_classes = []
testing_classes = []
testing = []

for i in range(len(model_data)):
    if "2018" in model_data[i][2]:
        training_features.append(model_data[i][0].lower())
        training_classes.append(model_data[i][1])
        training.append([model_data[i][0].lower(), model_data[i][1]])
    elif "2019" in model_data[i][2]:
        testing_features.append(model_data[i][0].lower())
        testing_classes.append(model_data[i][1])
        testing.append([model_data[i][0].lower(), model_data[i][1]])

vocabulary = {}
classes = {}
removed = string.punctuation + "“" + "”"
removed = removed.replace("-", "")
removed = removed.replace("_", "")

table = str.maketrans(dict.fromkeys(removed))
for i in training:
    if i[1] not in classes:
        classes[i[1]] = 0
    feature = i[0].translate(table)
    feature = feature.replace(" - ", "")
    feature = feature.encode("ascii", "ignore").decode("ascii")
    for word in feature.split():
        classes[i[1]] += 1
        if not (word in vocabulary):
            vocabulary[word] = {i[1] : [1, 0]}
        elif not (i[1] in vocabulary[word]):
            vocabulary[word][i[1]] = [1, 0]
        else:
            vocabulary[word][i[1]][0] += 1
            
for i in vocabulary.keys():
    for j in vocabulary[i]:
        vocabulary[i][j][1] = (vocabulary[i][j][0] + 0.5) / (classes[j] + 0.5 * len(vocabulary))

sorted_vocabulary = {}    
sorted_keys = sorted(vocabulary.keys(), key=lambda x:x)
for i in sorted_keys:
    sorted_vocabulary[i] = vocabulary[i]

model_2018 = open("model-2018.txt", "w")
buffer = []
line_counter = 0
for i in sorted_vocabulary.keys():
    buffer.append("{0}  {1}  ".format(line_counter, i))
    if "story" in sorted_vocabulary[i]:
        buffer.append("{0}  {1}  ".format(sorted_vocabulary[i]["story"][0], sorted_vocabulary[i]["story"][1]))
    if "ask_hn" in sorted_vocabulary[i]:
        buffer.append("{0}  {1}  ".format(sorted_vocabulary[i]["ask_hn"][0], sorted_vocabulary[i]["ask_hn"][1]))
    if "show_hn" in sorted_vocabulary[i]:
        buffer.append("{0}  {1}  ".format(sorted_vocabulary[i]["show_hn"][0], sorted_vocabulary[i]["show_hn"][1]))
    if "poll" in sorted_vocabulary[i]:
        buffer.append("{0}  {1}".format(sorted_vocabulary[i]["poll"][0], sorted_vocabulary[i]["poll"][1]))
    buffer.append("\n")
    model_2018.write("".join(buffer))
    buffer.clear()
    line_counter += 1
model_2018.close()

vocabulary_file = open("vocabulary.txt", "w")
for i in sorted_vocabulary.keys():
    vocabulary_file.write(i)
    vocabulary_file.write("\n")
model_2018.close()

print("Program Ended!")