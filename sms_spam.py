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
    # feature_set = [(wsd_features(inst), inst[3]) for inst in labeled_data]
    # train_set = feature_set[200:]
    # test_set = feature_set[:200]
    # return train_set, test_set
    pass


def wsd_features(instance):
    # context = instance[2]
    # position = instance[1]
    # preceding_item = context[position-1]
    # following_item = context[position+1]
    # synset_names = [[s.name() for s in wn.synsets(word)] if word.isalnum() else [None] for (word, tag) in context]
    # return {
    #     'preceding_word': preceding_item[0],
    #     'preceding_tag': preceding_item[1],
    #     'following_word': following_item[0],
    #     'following_tag': following_item[1],
    #     'name_in_sentence': ('NNP' in [tag for (word, tag) in context]),
    #     'preceding_synset_overlap': (not not set(synset_names[position]) & set(synset_names[position - 1])),
    #     'following_synset_overlap': (not not set(synset_names[position]) & set(synset_names[position + 1])),
    #     'sentence_length': len(context)
    #     }
    pass


def make_instance(tagged_sentence):
    # words = [t[0] for t in tagged_sentence]
    # position = words.index('interest')
    # return SensevalInstance('interest-n', position, tagged_sentence, [])
    pass


def train_classifier(training_set):
    # create the classifier
    # return nltk.NaiveBayesClassifier.train(training_set)
    pass


def evaluate_classifier(classifier, test_set):
    # get the accuracy and print it
    # print('classifier accuracy: ' + str(nltk.classify.accuracy(classifier, test_set)))
    pass


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
    # training_set, test_set = create_feature_sets(labeled_data)
    # classifier = train_classifier(training_set)
    # evaluate_classifier(classifier, test_set)
    # run_classifier(classifier)
