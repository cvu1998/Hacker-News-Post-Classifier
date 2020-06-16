import math
import string

import matplotlib.pyplot
import numpy
import pandas

def isInStopWord(word, stop_words):
    return word in stop_words

def isInbBetweenLength(word, bounds):
    return len(word) <= bounds[0] or len(word) >= bounds[1]

def getVocabulary(vocabulary, classes, training_set, table, exclude_function, *arg):
    for i in training:
        feature = i[0].translate(table)
        for word in feature.split():
            if exclude_function(word, *arg):
                continue
            classes[i[1]][0] += 1
            if not (word in vocabulary):
                vocabulary[word] = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}
                vocabulary[word][i[1]][0] += 1
            else:
                vocabulary[word][i[1]][0] += 1
                
def generateModel(vocabulary, sorted_vocabulary, classes, model_filename, write_to_file):
    for i in vocabulary.keys():
        for j in vocabulary[i]:
            vocabulary[i][j][1] = (vocabulary[i][j][0] + 0.5) / (classes[j][0] + 0.5 * len(vocabulary))
        
    for i in training_classes:
        if i in classes:
            classes[i][1] += 1
    
    for i in classes:
        classes[i][1] = classes[i][1] / len(training_classes)
    
    sorted_keys = sorted(vocabulary.keys(), key=lambda x:x)
    for i in sorted_keys:
        sorted_vocabulary[i] = vocabulary[i]
    
    if write_to_file:
        model = open(model_filename, "w", encoding="utf-8")
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
            model.write("".join(buffer))
            buffer.clear()
            line_counter += 1
        model.close()

def getScores(vocabulary, classes, feature, table):
    output = {}
    feature = feature.translate(table)
    for i in classes.keys():
        score = 0
        for word in feature.split():
            if word in vocabulary:
                if vocabulary[word][i][1] > 0:
                    score += math.log10(vocabulary[word][i][1])
                else:
                    score = float('-inf')
                    break
        if score > float("-inf") and classes[i][1] > 0:
            output[i] = score + math.log10(classes[i][1])
        else:
            output[i] = float("-inf")
    return output

def getMaxScoreClass(outputs):
    output = None
    max_score = float('-inf')
    for j in outputs.keys():
        score = outputs[j]
        if score >= max_score:
            output = j
            max_score = score
    return output

def generateResultAndFiles(vocabulary, classes, table, result_filename, inspect_filename, model_name, write_to_file):
    if write_to_file:
        result = open(result_filename, "w", encoding="utf-8")
        inspect_result = open(inspect_filename, "w", encoding="utf-8")
        buffer = []
    line_counter = 0
    model_score = 0
    for i in testing_features:
        outputs = getScores(vocabulary, classes, i, table)
        output = getMaxScoreClass(outputs)
        answer = testing_classes[line_counter]
        correctness = "wrong"
        if answer == output:
            correctness = "right"
            model_score += 1
        if write_to_file:
            result.write("{0}  {1}  {2} ".format(str(line_counter), i, output))
            if "story" in outputs:
                buffer.append("{0}  ".format(str(outputs["story"])))
            else:
                buffer.append("-inf  ")
            if "ask_hn" in outputs:
                buffer.append("{0}  ".format(str(outputs["ask_hn"])))
            else:
                buffer.append("-inf  ")
            if "show_hn" in outputs:
                buffer.append("{0}  ".format(str(outputs["show_hn"])))
            else:
                buffer.append("-inf  ")
            if "poll" in outputs:
                buffer.append("{0}  ".format(str(outputs["poll"])))
            else:
                buffer.append("-inf  ")
            result.write("".join(buffer))
            if correctness == "wrong":
                inspect_result.write(output + " " + answer + " " + "| " + i + "\n")
            result.write("{0}  {1}\n".format(answer, correctness))
            buffer.clear()
        line_counter += 1
    if write_to_file:
        result.close()
        inspect_result.close()
        
    print(model_name + ":")
    print("Accuracy: {0}".format(model_score / len(testing_features)))
    print("Accuracy: {0} / {1}".format(model_score, len(testing_features)))
    print("Wrongs: {0}".format(len(testing_features) - model_score))
    return model_score / len(testing_features)
    
def getWordsToBeRemovedBaseOnFrequency(vocabulary, frequency):
    words_to_pop = []
    for i in vocabulary.keys():
        if vocabulary[i]["story"][0] + vocabulary[i]["ask_hn"][0] + vocabulary[i]["show_hn"][0] + vocabulary[i]["poll"][0] <= frequency:
            words_to_pop.append(i)
    return words_to_pop

def sortFunction(i):
    return i[0]

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
sorted_vocabulary = {}
classes = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}

removed = string.punctuation + "“" + "”" + "«" + "—" + "" + "ð" + "–"
removed = removed.replace("-", "")
removed = removed.replace("_", "")

stop_words = set()

table = str.maketrans(dict.fromkeys(removed))

getVocabulary(vocabulary, classes, training, table, isInStopWord, stop_words)

percentile_based_voc = []
for i in vocabulary.keys():
    frequency = vocabulary[i]["story"][0] + vocabulary[i]["ask_hn"][0] + vocabulary[i]["show_hn"][0] + vocabulary[i]["poll"][0]
    percentile_based_voc.append([frequency, i])
    
generateModel(vocabulary, sorted_vocabulary, classes, "model-2018.txt", True)

vocabulary_file = open("vocabulary.txt", "w", encoding="utf-8")
for i in sorted_vocabulary.keys():
    vocabulary_file.write(i + "\n")
vocabulary_file.close()

generateResultAndFiles(vocabulary, classes, table, "baseline-result.txt", "inspect-result-basic.txt", "Basic Model", True)

stop_words_file = open("stopwords.txt", "r", encoding="utf-8")
for i in stop_words_file:
    i = i.replace("\n", "")
    stop_words.add(i)
stop_words_file.close()

vocabulary = {}
sorted_vocabulary = {}
classes = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}

getVocabulary(vocabulary, classes, training, table, isInStopWord, stop_words)

generateModel(vocabulary, sorted_vocabulary, classes, "stopword-model.txt", True)

generateResultAndFiles(vocabulary, classes, table, "stopword-result.txt", "inspect-result-stopword.txt", "Stopwords Model", True)

vocabulary = {}
sorted_vocabulary = {}
classes = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}

bounds = [2, 9]

getVocabulary(vocabulary, classes, training, table, isInbBetweenLength, bounds)

generateModel(vocabulary, sorted_vocabulary, classes, "wordlength-model.txt", True)

generateResultAndFiles(vocabulary, classes, table, "wordlength-result.txt", "inspect-result-wordlength.txt", "Word Length Model", True)

performances_frequency = []
numbers_of_words_frequency = []
frequencies = [1, 5, 10, 15, 20]
    
stop_words.clear()
for i in frequencies:
    vocabulary = {}
    sorted_vocabulary = {}
    classes = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}
    
    getVocabulary(vocabulary, classes, training, table, isInStopWord, stop_words)
    
    for j in getWordsToBeRemovedBaseOnFrequency(vocabulary, i):
        vocabulary.pop(j)
        
    generateModel(vocabulary, sorted_vocabulary, classes, "", False)

    numbers_of_words_frequency.append(len(vocabulary))
    
    performances_frequency.append(generateResultAndFiles(vocabulary, classes, table, "", "", "Frequency Model {0}".format(i), False))
        
percentile_based_voc.sort(reverse=True, key=sortFunction)

performances_percentile = []
numbers_of_words_percentile = []
percentiles = [0.05, 0.1, 0.15, 0.2, 0.25]

for i in percentiles:
    vocabulary = {}
    sorted_vocabulary = {}
    classes = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}
    
    getVocabulary(vocabulary, classes, training, table, isInStopWord, stop_words)
    
    for j in range(int(i * len(percentile_based_voc))):
        vocabulary.pop(percentile_based_voc[j][1])
    
    generateModel(vocabulary, sorted_vocabulary, classes, "", False)

    numbers_of_words_percentile.append(len(vocabulary))
    performances_percentile.append(generateResultAndFiles(vocabulary, classes, table, "", "", "Percentile Model {0}".format(i), False))

matplotlib.pyplot.figure(figsize=(4, 2))
matplotlib.pyplot.subplot(121)
matplotlib.pyplot.xlabel("Numbers of Words Left in the Vocabulary")
matplotlib.pyplot.ylabel("Accuracy of the Model")
matplotlib.pyplot.title("Frequency Model", loc="center")
matplotlib.pyplot.plot(numbers_of_words_frequency, performances_frequency, color='blue', marker='o')
matplotlib.pyplot.subplot(122)
matplotlib.pyplot.title("Percentile Model", loc="center")
matplotlib.pyplot.xlabel("Numbers of Words Left in the Vocabulary")
matplotlib.pyplot.ylabel("Accuracy of the Model")
matplotlib.pyplot.plot(numbers_of_words_percentile, performances_percentile, color='blue', marker='o')
matplotlib.pyplot.show()
print("Program Ended!")