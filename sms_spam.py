"""



"""

import os
import re
import random
import pprint

import nltk


def load_labeled_data(data_root):
    # create corpus and collect all labeled data
    data_dir = os.getcwd() + '\\' + data_root
    filename = 'sms_spam.csv'
    data = open(data_dir + '\\' + filename, encoding="Latin-1").readlines()
    data.pop(0)
    labeled_data = [(inst[:inst.find(',')], inst[inst.find(',')+1:len(inst)-1]) for inst in data]
    random.shuffle(labeled_data)
    return labeled_data


def create_feature_sets(labeled_data):
    # create feature sets
    feature_set = [(sms_features(inst), inst[0]) for inst in labeled_data]
    div = int(len(labeled_data) * 0.1)
    train_set = feature_set[div:]
    test_set = feature_set[:div]
    return train_set, test_set


def sms_features(instance):
    message = pre_process_data(instance[1])
    def has_slang(message):
    	slang = re.findall(r'(lol|lmao|wtf|bff|omg|rofl)')
    	return True if slang else False
    def has_emoticon(message):
    	emoticon = re.findall(r'[:;]\'?\s?(-?|\*?|\^)?\s?[\)\(DPp3O0o]')
    	return True if emoticon else False
    def is_spam_call(message):
        spam_call = re.findall(r'(txt|text|TXT|TEXT|call|CALL|Call)([A-Z]{,10}|[0-9]+)')
        return True if spam_call else False
    def is_over_length(message):
        return len(message)
    return {
        'message_length': len(message),
        'has_slang': has_slang(message),
        'has_emoticon': has_emoticon(message),
        'is_spam_call': is_spam_call(message),
        'length_of_message': is_over_length(message)
        }


def pre_process_data(message):
    # pre-processing of messages
    stopwords = nltk.corpus.stopwords.words('english')
    message = ' '.join(word for word in message.split() if word not in set(stopwords))
    return message


def train_classifier(training_set):
    # create the classifier
    return nltk.NaiveBayesClassifier.train(training_set)


def evaluate_classifier(classifier, test_set):
    # get the accuracy and print it
    print('classifier accuracy: ' + str(nltk.classify.accuracy(classifier, test_set)))


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
    pprint.pprint(labeled_data[:40])
    # training_set, test_set = create_feature_sets(labeled_data)
    # classifier = train_classifier(training_set)
    # evaluate_classifier(classifier, test_set)
    # run_classifier(classifier)
