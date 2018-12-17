"""
"""

import os
import re
import random
import pprint

import nltk

dir = os.getcwd()

def load_labeled_data(data_root):
    # create corpus and collect all labeled data
    data_dir = os.path.join(dir, data_root)
    filename = 'sms_spam.csv'
    data = open(os.path.join(data_dir, filename), encoding="Latin-1").read().splitlines()
    data.pop(0)
    labeled_data = [(inst[:inst.find(',')], pre_process_data(inst[inst.find(',')+1:])) for inst in data]
    random.shuffle(labeled_data)
    return labeled_data


def pre_process_data(message):
    # pre-processing of messages
    stopwords = nltk.corpus.stopwords.words('english')
    porter = nltk.PorterStemmer()
    message = ' '.join(porter.stem(word) for word in message.split() if word not in set(stopwords))
    return message


def create_feature_sets(labeled_data):
    # create feature sets
    ham_bigrams = [nltk.bigrams(text) for (label, text) in labeled_data if label == 'ham']
    spam_bigrams = [nltk.bigrams(text) for (label, text) in labeled_data if label == 'spam']
    feature_set = [(sms_features(inst, ham_bigrams, spam_bigrams), inst[0]) for inst in labeled_data]
    div = int(len(labeled_data) * 0.1)
    train_set = feature_set[div:]
    test_set = feature_set[:div]
    return train_set, test_set


def sms_features(instance, ham_bigrams, spam_bigrams):
    message = instance[1]
    message_tokens = nltk.word_tokenize(message)
    message_tags = nltk.pos_tag(message_tokens)
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
        'has_emoticon': re.findall(r'[:;]\'?\s?(-?|\*?|\^)?\s?[\)\(DPp3O0o]', message) is not None,
        'is_spam_call': re.findall(r'(txt|text|TXT|TEXT|call|CALL|Call)([A-Z]{,10}|[0-9]+)',message) is not None,
        'length_of_message': len_feature(message),
        'contains_gibberish': re.search(r'\b[A-z]+[0-9]+.*\b', message) is not None,
        'find_win': re.findall(r'win|won|winner', message, re.I) is not None,
        'find_urgent': re.findall(r'URGENT', message) is not None,
        'find_prize': re.findall(r'prize', message, re.I) is not None,
        'find_free': re.findall(r'FREE', message) is not None,
        'find_claim': re.findall(r'claim', message, re.I) is not None,
        'capitalization': re.findall(r'^[A-Z\s]+$', message) is not None,
        'find_website': re.findall(r'\A(http://)?(www)?.*\Z', message) is not None,
        'contains_name': [word for word in message_tokens if word.title() in name_data] != [],
        'contains_common_ham_bigram': [bgram for bgram in message_bigrams if bgram in fdist_ham_bigrams.most_common(10)] != [],
        'contains_common_spam_bigram': [bgram for bgram in message_bigrams if bgram in fdist_spam_bigrams.most_common(10)] != []
        }


def train_classifier_bayes(training_set):
    # create Naive Bayes classifier
    return nltk.NaiveBayesClassifier.train(training_set)


def train_classifier_dec_tree(training_set):
    # create decision tree classifier
    return nltk.DecisionTreeClassifier.train(training_set)


def train_classifier_max_ent(training_set):
    # create maximum entropy classifier
    return nltk.MaxentClassifier.train(training_set)


def evaluate_classifier(classifier, test_set):
    # get the accuracy of the test set
    return str(nltk.classify.accuracy(classifier, test_set))


def run_classifier(classifier):
    # get senseval data for Emma from Gutenberg corpus
    # any text from Gutenberg corpus can be subbed
    # prints random sentence that includes sense of 'interest'
    # emma = nltk.corpus.gutenberg.sents('austen-emma.txt')
    # tagged_sents = [nltk.pos_tag(s) for s in emma if 'interest' in s]
    # wsd_insts = [make_instance(s) for s in tagged_sents]
    # data = [(inst.word, inst.position, inst.context) for inst in wsd_insts]
    # output = []
    # for d in data:
    #     sense = classifier.classify(wsd_features(d))
    #     output.append([sense if word == 'interest' else word for (word, tag) in d[2]])
    # printable_output = [' '.join(sent) for sent in output]
    # print(random.choice(printable_output))
    pass


if __name__ == '__main__':

    labeled_data = load_labeled_data('SMS')
    training_set, test_set = create_feature_sets(labeled_data)
    classifier_bayes = train_classifier_bayes(training_set)
    classifier_dec_tree = train_classifier_dec_tree(training_set)
    classifier_max_ent = train_classifier_max_ent(training_set)
    print(classifier_dec_trees.pretty_format())
    nltk.NaiveBayesClassifier.train(training_set).show_most_informative_features(10)
    print('Naive Bayes accuracy ' + evaluate_classifier(classifier_bayes, test_set))
    print('Decision Tree accuracy ' + evaluate_classifier(classifier_dec_tree, test_set))
    print('Maximum Entropy accuracy ' + evaluate_classifier(classifier_max_ent, test_set))    
    # run_classifier(classifier)
