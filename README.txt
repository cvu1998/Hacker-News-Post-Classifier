Hacker-News-Post-Classifier

1. First, download the zip file A2_40061685.zip.
2. Extract the contents of the zip file.
3. Move your current directory to the directory where main.py is located.
4. Add a hns_2018_2019.csv file in the same directory as main.py, the row data for the csv file should follow so:
(Blank) Object ID, Title, Post Type ,Author, Created At, URL,Points, Number of Comments, Year
With posts created in year 2018 and 2019
5. Add a stopwords.txt file to the directory for a model using stop words.
6. To run the application, run main.py python script.
7. After running the program, as output, you should have model-2018.txt, stopword-model.txt, and wordlength-model.txt to view the 3 types of models.
8. After running the program, as output, you should have baseline-result.txt, stopword-result.txt, and wordlength-result.txt to view the 3 results of the models when applied the 2019 testing set from the csv file
9. Finally on the command prompt, you should be able to see accuracy of each model and view two plots with accuracy plotted against number of words remaining for frequency filtering and percentile filtering models