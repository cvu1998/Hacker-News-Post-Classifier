# -------------------------------------------------------
# Assignment 2
# Written by Cong-Vinh Vu, Student ID: 40061685
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

import math
import string

import matplotlib.pyplot
import numpy
import pandas

# Function used to include all words in the vocabulary
def skip(word, skip):
    return skip

# Function to exclude all stop words
def isInStopWord(word, stop_words):
    return word in stop_words

# Function to exclude all words outside input bounds
def isInbBetweenLength(word, bounds):
    return len(word) <= bounds[0] or len(word) >= bounds[1]

# Function to generate a vocabulary given a training set and exclude function
def getVocabulary(vocabulary, training_set, table, exclude_function, *arg):
    # Training list, index 0 gives the title, index 1 gives the class
    for i in training:
        # Remove punctuation from the title
        feature = i[0].translate(table)
        for word in feature.split():
            if exclude_function(word, *arg):
                continue
            # Add the word to the vocabulary, the key is the word and increment it's frequency for that class
            # Index [0] gives the frequency of the word for a specific class, index 1 gives the probability of the word for a specific class
            if not (word in vocabulary):
                vocabulary[word] = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}
                vocabulary[word][i[1]][0] += 1
            else:
                vocabulary[word][i[1]][0] += 1
                
# Function to generate number of words per class 
# Classes dictionnary, index 0 contains number of words per class, index 1 contains probability of each classes                
def generateWordCountInClasses(vocabulary, classes):
    # Add word count for every class by looking up the vocabulary dictionnary
    for j in vocabulary.keys():
        # classes dictionnary, index 0 contains number of words per class, index 1 contains probability of each classes
        classes["story"][0] += vocabulary[j]["story"][0]
        classes["ask_hn"][0] += vocabulary[j]["ask_hn"][0]
        classes["show_hn"][0] += vocabulary[j]["show_hn"][0]
        classes["poll"][0] += vocabulary[j]["poll"][0]
   
# Function to generate a model with model file if wanted             
def generateModel(vocabulary, sorted_vocabulary, classes, model_filename, write_to_file):
    # Generate the probabilities for each word in the vocabulary depending on the class
    for i in vocabulary.keys():
        for j in vocabulary[i]:
            vocabulary[i][j][1] = (vocabulary[i][j][0] + 0.5) / (classes[j][0] + 0.5 * len(vocabulary))
       
    # Generate the number of class
    for i in training_classes:
        if i in classes:
            classes[i][1] += 1
    
    # Generate probability for each class
    for i in classes:
        classes[i][1] = classes[i][1] / len(training_classes)
    
    # Generate a sorted vocabulary
    sorted_keys = sorted(vocabulary.keys(), key=lambda x:x)
    for i in sorted_keys:
        sorted_vocabulary[i] = vocabulary[i]
    
    # Write the model to a text file if specified
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

# Get the score for a given title, for all classes
def getScores(vocabulary, classes, feature, table):
    output = {}
    # Remove punctutation from the feature
    feature = feature.translate(table)
    # For loop the get the score in log10 for a given title
    for i in classes.keys():
        score = 0
        for word in feature.split():
            if word in vocabulary:
                score += math.log10(vocabulary[word][i][1])
        # If they are no words in a class, the score is -infinity
        if classes[i][1] > 0:
            output[i] = score + math.log10(classes[i][1])
        else:
            output[i] = float("-inf")
    return output

# Return the class with the highest score
def getMaxScoreClass(outputs):
    output = None
    max_score = float('-inf')
    for j in outputs.keys():
        score = outputs[j]
        if score >= max_score:
            output = j
            max_score = score
    return output

# Function used to generate th result files as well as print the accuracy of the model given a testing set
def generateResultAndFiles(vocabulary, classes, table, result_filename, inspect_filename, model_name, write_to_file):
    if write_to_file:
        result = open(result_filename, "w", encoding="utf-8")
        inspect_result = open(inspect_filename, "w", encoding="utf-8")
        buffer = []
    line_counter = 0
    model_score = 0
    # Get the scores for every title in the testing set, as well as the most likely class, and compare with the correct class
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
    print("Wrongs: {0}\n".format(len(testing_features) - model_score))
    return model_score / len(testing_features)

# Function used to get words to be removed from the vocabulary given a frequency    
def getWordsToBeRemovedBaseOnFrequency(vocabulary, frequency):
    words_to_pop = []
    # Find words with overall frequency lower than input and add them to a list to be to be popped later on
    for i in vocabulary.keys():
        if vocabulary[i]["story"][0] + vocabulary[i]["ask_hn"][0] + vocabulary[i]["show_hn"][0] + vocabulary[i]["poll"][0] <= frequency:
            words_to_pop.append(i)
    return words_to_pop

def sortFunction(i):
    return i[0]

# Get data from the csv file, with columns 2 (Titles), 3 (Classes), and 5 (Dates)
dataset = pandas.read_csv("hns_2018_2019.csv")
data = dataset.iloc[:, :].values
model_data = dataset.iloc[:, [2, 3, 5]].values

training_features = []
testing_features = []
training = []

training_classes = []
testing_classes = []
testing = []

# Seperate training and testing set
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

# Punctuation to be removed
removed = string.punctuation + "“" + "”" + "«" + "—" + "" + "ð" + "–"
removed = removed.replace("-", "")
removed = removed.replace("_", "")

stop_words = set()

table = str.maketrans(dict.fromkeys(removed))

# Generate basic vocabulary with no words removed
getVocabulary(vocabulary, training, table, skip, False)

# Generate word count for each class
generateWordCountInClasses(vocabulary, classes)

# Create a list containing the word and it's overall frequency, index 0 represents the the word, index 1 gives its' overall frequency
percentile_based_voc = []
for i in vocabulary.keys():
    frequency = vocabulary[i]["story"][0] + vocabulary[i]["ask_hn"][0] + vocabulary[i]["show_hn"][0] + vocabulary[i]["poll"][0]
    percentile_based_voc.append([frequency, i])
   
# Sort the list, with higher frequency first
percentile_based_voc.sort(reverse=True, key=sortFunction)
    
# Generate the basic model
generateModel(vocabulary, sorted_vocabulary, classes, "model-2018.txt", True)

# Write the alphabetically sorted vocabulary to a file
vocabulary_file = open("vocabulary.txt", "w", encoding="utf-8")
for i in sorted_vocabulary.keys():
    vocabulary_file.write(i + "\n")
vocabulary_file.close()

# Generate baseline result files, inspect-result file is used to inspect which title did the model predict in a wrong manner
generateResultAndFiles(vocabulary, classes, table, "baseline-result.txt", "inspect-result-basic.txt", "Basic Model", True)

# Get the stop words and ass them to a set, to be removed using isInStopWord function
stop_words_file = open("stopwords.txt", "r", encoding="utf-8")
for i in stop_words_file:
    i = i.replace("\n", "")
    stop_words.add(i)
stop_words_file.close()

vocabulary = {}
sorted_vocabulary = {}
classes = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}

# Get the vocabulary for the stop word model, excluding stop words using isInStopWord function
getVocabulary(vocabulary, training, table, isInStopWord, stop_words)

# Generate word count for each class
generateWordCountInClasses(vocabulary, classes)

# Generate the model and write it to stopword-model.txt
generateModel(vocabulary, sorted_vocabulary, classes, "stopword-model.txt", True)

# Generate result files for stop-word model
generateResultAndFiles(vocabulary, classes, table, "stopword-result.txt", "inspect-result-stopword.txt", "Stopwords Model", True)

vocabulary = {}
sorted_vocabulary = {}
classes = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}

bounds = [2, 9]

# Get the vocabulary for the word-length model, excluding words outside length 2 and 9 using isInbBetweenLength function
getVocabulary(vocabulary, training, table, isInbBetweenLength, bounds)

# Generate word count for each class
generateWordCountInClasses(vocabulary, classes)

# Generate the model and write it to wordlength-model.txt
generateModel(vocabulary, sorted_vocabulary, classes, "wordlength-model.txt", True)

# Generate result files for word-length model
generateResultAndFiles(vocabulary, classes, table, "wordlength-result.txt", "inspect-result-wordlength.txt", "Word Length Model", True)

performances_frequency = []
numbers_of_words_frequency = []
# Frequencies to be excluded per model
frequencies = [1, 5, 10, 15, 20]

for i in frequencies:
    # Reset vocabulary and classes dictionnary
    vocabulary = {}
    sorted_vocabulary = {}
    classes = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}
    
    # Generate new basic vocabulary
    getVocabulary(vocabulary, training, table, skip, False)
    
    # Get the words to be removed using function getWordsToBeRemovedBaseOnFrequency
    # Remove words with lower frequency than i from the vocabulary
    for j in getWordsToBeRemovedBaseOnFrequency(vocabulary, i):
        vocabulary.pop(j)
        
    # Generate word count for each class
    generateWordCountInClasses(vocabulary, classes)
        
    # Generate the new model
    generateModel(vocabulary, sorted_vocabulary, classes, "", False)

    # Append number of words remaining in vocabulary for x axis 
    numbers_of_words_frequency.append(len(vocabulary))
    # Append accuracy of the model for y axis
    performances_frequency.append(generateResultAndFiles(vocabulary, classes, table, "", "", "Frequency Model {0}".format(i), False))

performances_percentile = []
numbers_of_words_percentile = []
# Percentiles of frequency to be removed
percentiles = [0.05, 0.1, 0.15, 0.2, 0.25]

for i in percentiles:
    # Reset vocabulary and classes dictionnary
    vocabulary = {}
    sorted_vocabulary = {}
    classes = {"story" : [0, 0], "ask_hn" : [0, 0], "show_hn" : [0, 0], "poll" : [0, 0]}
    
    # Generate new basic vocabulary
    getVocabulary(vocabulary, training, table, skip, False)
    
    # Remove the top i most frequent word in the vocabulary
    for j in range(int(i * len(percentile_based_voc))):
        vocabulary.pop(percentile_based_voc[j][1])
        
    # Generate word count for each class
    generateWordCountInClasses(vocabulary, classes)  
    
     # Generate the new model
    generateModel(vocabulary, sorted_vocabulary, classes, "", False)

    # Append number of words remaining in vocabulary for x axis
    numbers_of_words_percentile.append(len(vocabulary))
    # Append accuracy of the model for y axis
    performances_percentile.append(generateResultAndFiles(vocabulary, classes, table, "", "", "Percentile Model {0}".format(i), False))

# Plot the data for frequency and percentile models as subplots
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