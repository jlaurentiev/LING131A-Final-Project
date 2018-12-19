# LING131A-Final-Project
Spam filter final project repository for Intro to NLP

SMS data source: https://www.kaggle.com/uciml/sms-spam-collection-dataset/home

Names data source: https://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/

Code - who wrote what:

load_labeled_data -- John;
pre_process_data -- John;
create_feature_sets -- John;
sms_features -- Mingfan, Sarah, John, Meiqi;
  Mingfan - find_award, find_website, length_of_message, is_spam_call
  Sarah - has_slang, has_emoticons
  John - contains_gibberish, contains_name, contains_common_ham_bigram, contains_common_spam_bigram
  Meiqi - find_win/urgent/prize/free/claim, capitalization
get_ham_bigrams -- John;
get_spam_bigrams -- John;
train_classifier_bayes -- John;
train_classifier_max_ent -- John;
train_classifier_dec_tree -- Sarah;
evaluate_classifier -- Sarah;
run_classifier -- Meiqi;
confusion_matrix -- John;
