import csv
import sys

import nltk
from nltk.classify import NaiveBayesClassifier, util
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import SVC

reload(sys)
sys.setdefaultencoding('utf8')

# Open the training.txt file
data_file = open('training.txt')

# Read the training.txt file as a csv with tab as the delimiter.
# The list method converts the tuple returned by the csv.reader to a lost
training_data = list(csv.reader(data_file, delimiter='\t'))


def create_and_evaluate_classifier_10_fold(features):

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

    features = []
    for record in records:
        tokens = create_tokens(record[1])
        no_stopword_tokens = remove_stopwords(tokens)

        unigram_features = {token: True for token in no_stopword_tokens}
        features.append([unigram_features, int(record[0])])
    return features


def create_uni_and_bigram_features(records):

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

    features = []
    for record in records:
        tokens = create_tokens(record[1])
        no_stopword_tokens = remove_stopwords(tokens)

        bigrams = nltk.ngrams(no_stopword_tokens, 2)
        bigram_features = {bigram: True for bigram in bigrams}
        features.append([bigram_features, int(record[0])])

    return features


def create_pos_tag_features(records):

    features = []
    for record in records:
        tokens = create_tokens(record[1])
        post_tags = create_pos_tags(tokens)

        unigram_features = {token: True for token in post_tags}
        features.append([unigram_features, int(record[0])])

    return features


# Creates tokens, including case conversion
def create_tokens(text):
    return word_tokenize(text.lower())


# Remove english stopword tokens
def remove_stopwords(tokens):
    stopword_set = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stopword_set]
    return tokens


# Filter duplicates in data set
def filter_duplicates(records):
    tuples = (tuple(record) for record in records)
    return [list(record) for record in set(tuples)]


def merge_two_dicts(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict


def print_features(features):
    print (features)
    with open('features.json', 'w') as features_file:
        for feature in features:
            features_file.write("%s\n" % feature)


def count_pos_and_neg_entries(records):
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
    return nltk.pos_tag(tokens)


def run_tests():
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
    run_tests()


if __name__ == '__main__':
    main()
