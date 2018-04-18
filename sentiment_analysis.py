import csv
import sys

import nltk
from nltk.classify import NaiveBayesClassifier, util
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import KFold

reload(sys)
sys.setdefaultencoding('utf8')


def create_and_evaluate_classifier_10_fold(features):
    """
    Uses 10 fold cross validation to create a classifier with naive bayes
    for the given features. Evaluates the created classifier afterwards.
    Results are printed to the console
    :param features: List of features
    """

    k_fold_validator = KFold(n_splits=10, shuffle=False)
    accuracies = []
    counter = 1

    for train_index, test_index in k_fold_validator.split(features):
        # print str(counter * 10) + '% cross-validation'

        # Split features in training and test set
        training_features = [features[i] for i in train_index]
        test_features = [features[i] for i in test_index]

        # Create classifier
        classifier = NaiveBayesClassifier.train(training_features)

        # Evaluate classifier
        accuracy = util.accuracy(classifier, test_features)

        accuracies.append(accuracy)
        counter = counter + 1

    print 'Accuracy:', sum(accuracies) / float(len(accuracies)), '\n'


def create_unigram_features(records):
    """
    Creates unigram features from list of records for classification
    :param records: List of records
    :return: List of features
    """
    features = []
    for record in records:
        tokens = create_tokens(record[1])
        no_stopword_tokens = remove_stopwords(tokens)

        unigram_features = {token: True for token in no_stopword_tokens}
        features.append([unigram_features, int(record[0])])
    return features


def create_uni_and_bigram_features(records):
    """
    Creates unigram and bigram features from list of records for classification
    :param records: List of records
    :return: List of features
    """
    features = []
    for record in records:
        tokens = create_tokens(record[1])
        no_stopword_tokens = remove_stopwords(tokens)

        unigram_features = {token: True for token in no_stopword_tokens}

        bigrams = nltk.ngrams(no_stopword_tokens, 2)
        bigram_features = {bigram: True for bigram in bigrams}

        features.append([merge_two_dicts(unigram_features, bigram_features), int(record[0])])

    return features


def create_bigram_features(records):
    """
    Creates bigram features from list of records to be used for classification
    :param records: List of records
    :return: List of features
    """

    features = []
    for record in records:
        tokens = create_tokens(record[1])
        no_stopword_tokens = remove_stopwords(tokens)

        bigrams = nltk.ngrams(no_stopword_tokens, 2)
        bigram_features = {bigram: True for bigram in bigrams}
        features.append([bigram_features, int(record[0])])

    return features


def create_pos_tag_features(records):
    """
    Create part of speech tag features from records using nltk to be used for classification
    :param records: List of records
    :return: List of pos tag features
    """
    features = []
    for record in records:
        tokens = create_tokens(record[1])
        post_tags = create_pos_tags(tokens)

        unigram_features = {token: True for token in post_tags}
        features.append([unigram_features, int(record[0])])

    return features


# Creates tokens, including case conversion
def create_tokens(text):
    """
    Creates tokens from a given text
    :param text: Text to tokenize
    :return: List of tokens
    """
    return word_tokenize(text.lower())


# Remove english stopword tokens
def remove_stopwords(tokens):
    """
    Removes stopwords from list of tokens
    :param tokens: List of tokens to remove stopwords from
    :return: List of tokens without stopwords
    """
    stopword_set = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stopword_set]
    return tokens


# Filter duplicates in data set
def filter_duplicates(records):
    """
    Filters duplicates records
    :param records: List of records to be filtered
    :return: Filtered records
    """
    tuples = (tuple(record) for record in records)
    return [list(record) for record in set(tuples)]


def merge_two_dicts(dict1, dict2):
    """
    Merges two dictionaries into one
    :param dict1: First dictionary to merge
    :param dict2: Second dictionary to merge
    :return: Merged dictionary
    """
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict


def print_features(features):
    """
    Helper function that prints the given features to a features.json file
    :param features: List of features
    """
    print (features)
    with open('features.json', 'w') as features_file:
        for feature in features:
            features_file.write("%s\n" % feature)


def count_pos_and_neg_entries(records):
    """
    Counts and prints the number of positive and negative records in the given data
    :param records List of records that should be evaluated
    """
    pos = 0
    neg = 0
    for record in records:
        if record[0] == '1':
            pos = pos + 1
        else:
            neg = neg + 1

    print 'pos', pos
    print 'neg', neg


def create_pos_tags(tokens):
    """
    Creates part of speech tags from the given tokens
    :param tokens: List of tokens
    :return: List of pos tags
    """
    return nltk.pos_tag(tokens)


def get_training_data_from_file():
    """
    Reads the training.txt file with the csv reader
    :returns: Training data from the file training.txt
    """

    data_file = open('training.txt')
    training_data = list(csv.reader(data_file, delimiter='\t'))
    return training_data


def run_tests():
    """
    Run tests for four different features selection methods.
    Prints the results for each test to the console.
    """
    training_data = get_training_data_from_file()
    filtered_records = filter_duplicates(training_data)
    count_pos_and_neg_entries(filtered_records)

    unigram_features = create_unigram_features(filtered_records)
    uni_and_bigram_features = create_uni_and_bigram_features(filtered_records)
    bigram_features = create_bigram_features(filtered_records)
    pos_tag_features = create_pos_tag_features(filtered_records)

    print 'Train and evaluate unigram features:'
    create_and_evaluate_classifier_10_fold(unigram_features)

    print 'Train and evaluate on unigram and bigram features:'
    create_and_evaluate_classifier_10_fold(bigram_features)

    print 'Train and evaluate on bigram features'
    create_and_evaluate_classifier_10_fold(uni_and_bigram_features)

    print 'Train and evaluate on pos tag features'
    create_and_evaluate_classifier_10_fold(pos_tag_features)


def main():
    """Entry point of system"""
    run_tests()


if __name__ == '__main__':
    main()
