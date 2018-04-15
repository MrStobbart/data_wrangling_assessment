from sklearn import datasets, svm
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import wordpunct_tokenize, word_tokenize
import csv
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
import json

reload(sys)
sys.setdefaultencoding('utf8')

# Open the training.txt file
data_file = open('training.txt')

# Read the training.txt file as a csv with tab as the delimiter.
# The list method converts the tuple returned by the csv.reader to a lost
training_data = list(csv.reader(data_file, delimiter='\t'))


def create_classifier(features):
    set_split = int(len(features) * 0.9)
    training_set = features[:set_split]
    test_set = features[set_split:]

    print'train on %d instances, test on %d instances' % (len(training_set), len(test_set))
    classifier = NaiveBayesClassifier.train(training_set)
    print 'accuracy:', nltk.classify.util.accuracy(classifier, test_set)


def create_classifier_10_fold(features):
    k_fold_validator = KFold(n_splits=10, shuffle=True)

    accuracies = []

    for train_index, test_index in k_fold_validator.split(features):
        train_features = [features[i] for i in train_index]
        test_features = [features[i] for i in test_index]

        print'train on %d instances, test on %d instances' % (len(train_features), len(test_features))
        classifier = NaiveBayesClassifier.train(train_features)

        accuracy = nltk.classify.util.accuracy(classifier, test_features)
        accuracies.append(accuracy)

    print 'Accuracy 10 fold: ', sum(accuracies) / len(accuracies)


def create_features(records):
    # TODO Do this after after pre-processing
    filtered_records = filter_duplicates(records)

    pos = 0
    neg = 0
    for record in filtered_records:
        if record[0] == '1':
            pos = pos + 1
        else:
            neg = neg + 1

    print 'pos', pos
    print 'neg', neg

    features = []
    for record in filtered_records:
        tokens = create_tokens(record[1])
        no_stopword_tokens = remove_stopwords(tokens)

        unigram_features = {token: True for token in no_stopword_tokens}

        bigrams = nltk.ngrams(no_stopword_tokens, 2)
        bigram_features = {bigram: True for bigram in bigrams}

        # features.append([unigram_features, int(record[0])])
        features.append([merge_two_dicts(unigram_features, bigram_features), int(record[0])])

    return features


# Creates tokens, including case conversion
def create_tokens(text):
    return wordpunct_tokenize(text.lower())


# Remove english stopword tokens
def remove_stopwords(tokens):
    stopword_set = set(stopwords.words('english'))
    return [token for token in tokens if token not in stopword_set]


# Filter duplicates in data set
def filter_duplicates(records):
    tuples = (tuple(record) for record in records)
    return [list(record) for record in set(tuples)]


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def print_features(features):
    print (features)
    with open('features.json', 'w') as file:
        for feature in features:
            file.write("%s\n" % feature)


features = create_features(training_data)
create_classifier(features)
create_classifier_10_fold(features)
