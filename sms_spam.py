"""
"""

import os
import re
import random

import nltk

dir = os.getcwd()


def load_labeled_data(data_root, filename):
    # create corpus and collect all labeled data
    data_dir = os.path.join(dir, data_root)
    data = open(os.path.join(data_dir, filename), encoding="Latin-1").read().splitlines()
    data.pop(0)
    labeled_data = [(inst[:inst.find(',')], pre_process_data(inst[inst.find(',')+1:])) for inst in data]
    # random.shuffle(labeled_data)
    return labeled_data


def pre_process_data(message):
    # pre-processing of messages
    stopwords = nltk.corpus.stopwords.words('english')
    porter = nltk.PorterStemmer()
    message = ' '.join(porter.stem(word) for word in message.split() if word not in set(stopwords))
    return message


def create_feature_sets(labeled_data):
    # create feature sets
    ham_bigrams = get_ham_bigrams(labeled_data)
    spam_bigrams = get_spam_bigrams(labeled_data)
    feature_set = [(sms_features(inst, ham_bigrams, spam_bigrams), inst[0]) for inst in labeled_data]
    div = int(len(labeled_data) * 0.1)
    train_set = feature_set[div:]
    dev_test_set = feature_set[:div]
    return train_set, dev_test_set


def sms_features(instance, ham_bigrams, spam_bigrams):
    # get features for each instance
    message = instance[1]
    message_tokens = nltk.word_tokenize(message)
    name_data = open(os.path.join(dir, 'Names.txt')).read().splitlines()
    fdist_ham_bigrams = nltk.FreqDist(ham_bigrams)
    fdist_spam_bigrams = nltk.FreqDist(spam_bigrams)
    message_bigrams = list(nltk.bigrams(message_tokens))

    def len_feature(text):
        if len(text) > 200:
            return 'long'
        elif len(text) > 85:
            return 'medium'
        else:
            return 'short'
    return {
        'has_slang': re.search(r'(lol|lmao|wtf|bff|omg|rofl)', message) is not None,
        'has_emoticon': re.search(r'[:;]\'?\s?(-?|\*?|\^)?\s?[\)\(DPp3O0o]', message) is not None,
        'is_spam_call': re.search(r'(txt|text|TXT|TEXT|call|CALL|Call)([A-Z]{,10}|[0-9]+)', message) is not None,
        'length_of_message': len_feature(message),
        'contains_gibberish': re.search(r'\b[A-z]+[0-9]+.*\b', message) is not None,
        'find_win': re.search(r'(win|won|winner)', message, re.I) is not None,
        'find_award': re.search(r'award', message, re.I) is not None,
        'find_urgent': re.search(r'(Urgent|URGENT)', message) is not None,
        'find_prize': re.search(r'prize', message, re.I) is not None,
        'find_free': re.search(r'FREE', message) is not None,
        'find_claim': re.search(r'claim', message, re.I) is not None,
        'capitalization': re.search(r'^[A-Z\s]+$', message) is not None,
        'find_website': re.search(r'www', message) is not None,
        'contains_name': [word for word in message_tokens if word.title() in name_data] != [],
        'contains_common_ham_bigram': [bgram for bgram in message_bigrams if bgram in fdist_ham_bigrams.most_common(10)] != [],
        'contains_common_spam_bigram': [bgram for bgram in message_bigrams if bgram in fdist_spam_bigrams.most_common(10)] != []
        }


def get_ham_bigrams(labeled_data):
    # get bigrams for ham data
    return [nltk.bigrams(text) for (label, text) in labeled_data if label == 'ham']


def get_spam_bigrams(labeled_data):
    # get bigrams for spam data
    return [nltk.bigrams(text) for (label, text) in labeled_data if label == 'spam']


def train_classifier_bayes(training_set):
    # create naive Bayes classifier
    return nltk.NaiveBayesClassifier.train(training_set)


def train_classifier_dec_tree(training_set):
    # create decision tree classifier
    return nltk.DecisionTreeClassifier.train(training_set)


def train_classifier_max_ent(training_set):
    # create maximum entropy classifier
    return nltk.MaxentClassifier.train(training_set)


def evaluate_classifier(classifier, dev_test_set):
    # get the accuracy of the test set
    return str(nltk.classify.accuracy(classifier, dev_test_set))


def run_classifier(classifier):
    # returns accuracy of classifier on test data
    labeled_data = load_labeled_data('SMS', 'test_data.csv')
    ham_bigrams = get_ham_bigrams(labeled_data)
    spam_bigrams = get_spam_bigrams(labeled_data)
    test_set = [(sms_features(inst, ham_bigrams, spam_bigrams), inst[0]) for inst in labeled_data]
    return evaluate_classifier(classifier, test_set)


def confusion_matrix(classifier):
    # prints the confusion matrix on test data using classifier
    data_dir = os.path.join(dir, 'SMS')
    filename = 'test_data.csv'
    data = open(os.path.join(data_dir, filename), encoding="Latin-1").read().splitlines()
    labeled_data = [(inst[:inst.find(',')], pre_process_data(inst[inst.find(',')+1:])) for inst in data]
    ham_bigrams = get_ham_bigrams(labeled_data)
    spam_bigrams = get_spam_bigrams(labeled_data)
    gold = [label for (label, message) in labeled_data]
    test = [classifier.classify(sms_features(inst, ham_bigrams, spam_bigrams)) for inst in labeled_data]
    cm = nltk.ConfusionMatrix(gold, test)
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))


if __name__ == '__main__':

    labeled_data = load_labeled_data('SMS','development_data.csv')
    training_set, dev_test_set = create_feature_sets(labeled_data)
    classifier_bayes = train_classifier_bayes(training_set)
    confusion_matrix(classifier_bayes)
    # classifier_dec_tree = train_classifier_dec_tree(training_set)
    # confusion_matrix(classifier_dec_tree)
    # classifier_max_ent = train_classifier_max_ent(training_set)
    # confusion_matrix(classifier_max_ent)
    # print(classifier_dec_tree.pretty_format())
    # nltk.NaiveBayesClassifier.train(training_set).show_most_informative_features(10)
    # nltk.MaxentClassifier.train(training_set).show_most_informative_features(10)
    # print('Naive Bayes accuracy ' + evaluate_classifier(classifier_bayes, dev_test_set))
    # print('Decision Tree accuracy ' + evaluate_classifier(classifier_dec_tree, dev_test_set))
    # print('Maximum Entropy accuracy ' + evaluate_classifier(classifier_max_ent, dev_test_set))    
    # run_classifier(classifier) #run it on classifier_bayes, classifier_dec_tree and classifier_max_ent
