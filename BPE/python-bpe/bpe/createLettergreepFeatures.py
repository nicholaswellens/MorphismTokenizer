#from encoder import Encoder
import datetime
import time
from polyglot.text import Text, Word
import ucto
import os

import IPython


from morphism_encoder_merge_lettergreep import Encoder
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from frog import Frog, FrogOptions
from numpy import asarray, savetxt, loadtxt
from string import punctuation

########################################

########## HELPER FUNCTIONS ############

start_time = time.time()

def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''
  # Round to the nearest second.
  elapsed_rounded = int(round((elapsed)))

  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))



#Deleting of \n in reviews, maybe can use for sentence tag putting.
#Misschien wel goed zo, maar als je vergelijkt met word-tokenization misschien ook verwijderen in de google colab

def change_text_to_morphs(sentences):
    # sentence List to sentence list in morphism form
    morphSentences = []

    frog = Frog(
        FrogOptions(tok=True, lemma=True, morph=True, daringmorph=False, mwu=False, chunking=False, ner=False,
                    parser=False))

    for sentenceNumber in range(0,len(sentences)):
        print(sentenceNumber)
        print("of")
        print(len(sentences))
        sentenceToBeProcessed = sentences[sentenceNumber]
        sentenceToBeProcessed = sentenceToBeProcessed.replace("\n"," ")
        morphSentence = []
        output = frog.process(sentenceToBeProcessed)
        for i in range(0,len(output)):
            morphisms_word = output[i].get("morph")
            morphisms_word_list = morphisms_word.replace('[', '').split(']')
            #print("EVET")
            #print(morphisms_word_list)
            morphSentence += morphisms_word_list
        #print("MORPHSENTENCE")
        #print(morphSentence)
        # Remove the empty strings
        morphSentence = list(filter(None, morphSentence))
        morphSentence = ' '.join(morphSentence)
        #print("HERE")
        #print(morphSentence)
        morphSentences += [morphSentence]

    #with open('MorphedTestReviews.pickle', 'wb') as outputfile:
    #    pickle.dump(morphSentences, outputfile)
    return morphSentences

##########################################################

configurationFile = "tokconfig-nld"
tokenizer = ucto.Tokenizer(configurationFile)

def ucto_tokenize(sentence):

    tokenized_sentence = []
    tokenizer.process(sentence)
    for token in tokenizer:
      tokenized_sentence += [str(token)]
    return tokenized_sentence

#def convertToSelfTrainedMorf(sentences, train_file, convert_file):
#    #List of str to List of str (morfemes)

#    os.system("morfessor -t " + str(train_file) + " -S model.segm -T " + str(convert_file))
#     os.system("morfessor -t EenPosReview.txt -S model.segm -T EenPosReview.txt")

# print("here")
# convertToSelfTrainedMorf([],"EenPosReview.txt","EenPosReview.txt")
# print("now here")
def convertToPolyglotMorf(sentences, save = False):
    #List of str to List of str (morfemes)

    total_review_morf_text_list = []
    i = 1
    morfed_sentences = []
    print(len(sentences))
    for sentence in sentences:
      print(i)
      tokenized_sentence = ucto_tokenize(sentence)
      morfed_sentence = []
      for w in tokenized_sentence:
        w = str(w)
        w = Word(w, language="nl")
        #print("{:<20}{}".format(w, w.morphemes))
        morfed_sentence += w.morphemes
      #print(review_morf_list)
      morfed_sentences += morfed_sentence
      i+=1

      morfed_sentences_text = '*%*%'.join(morfed_sentences)

    if save is True:

        with open("convertedPolyglotMorfText.txt", "w") as text_file:
            text_file.write(morfed_sentences_text)

    return morfed_sentences


########## VARIABLES ##########

fixed_len = 25
split_frac = 0.9 #splitting fraction between TRAIN and TEST data


#########################################

data = pd.read_csv("lettergrepen_1.csv")

print(data["woorden"])
print(data["lettergrepen"])
print(data["aantal lettergrepen"])

lettergreep_words = list(data["woorden"])
lettergreep_splits = list(data["lettergrepen"])
lettergreep_labels = list(data["aantal lettergrepen"])




len_dataset = len(lettergreep_words)

lettergreep_train_words = lettergreep_words[0:int(split_frac*len_dataset)]
lettergreep_train_labels = lettergreep_labels[0:int(split_frac*len_dataset)]

lettergreep_test_words = lettergreep_words[int(split_frac*len_dataset):]
lettergreep_test_labels = lettergreep_labels[int(split_frac*len_dataset):]

#IPython ; IPython.embed() ; exit(1)





########## LOAD AND SHUFFLE DATA, CREATE LABELS ##########

### Read Shuffled data and labels ###

print("loadedStart")

with open('totTrainListShuffled.data', 'rb') as filehandle:
      # read the data as binary data stream
      totTrainList = pickle.load(filehandle)

#print("loaded")

with open('trainLabelListShuffled.data', 'rb') as filehandle:
      # read the data as binary data stream
      trainLabelList = pickle.load(filehandle)

print("loadedd")

with open('totTestListShuffled.data', 'rb') as filehandle:
      # read the data as binary data stream
      totTestList = pickle.load(filehandle)

print("loadeddd")

with open('testLabelListShuffled.data', 'rb') as filehandle:
      # read the data as binary data stream
      testLabelList = pickle.load(filehandle)

print("loadedddd")
#
# with open('MorphedTrainReviews.pickle', 'rb') as inputfile:
#     MorphedTrainReviewsList = pickle.load(inputfile)
#
# print("loadeddddd")
#
# with open('MorphedTestReviews.pickle', 'rb') as inputfile:
#     MorphedTestReviewsList = pickle.load(inputfile)
#
# print("loadedddddd")
#
# with open('all_text2_morf_NLTK.pickle', 'rb') as inputfile:
#     all_text2_morf_NLTK = pickle.load(inputfile)
#
# a_list_of_all_text2_morf_NLTK = [all_text2_morf_NLTK]

# print("loadeddddddd")

# with open('total_review_morf_list_text.pickle', 'rb') as inputfile:
#     MorfedUctoTrainReviewsText = pickle.load(inputfile)
#     MorfedUctoTrainReviewsList = MorfedUctoTrainReviewsText.split("*$*$*$*$")

# print("loadedddddddd")

#UctoOpen = open('UctoMorfessorSelfTrained.txt','r')
#SelfTrainedMorfedUctoTrainReviewsList = UctoOpen.read()
#SelfTrainedMorfedUctoTrainReviewsList = [SelfTrainedMorfedUctoTrainReviewsList]

#with open('total_review_word_list_text.pickle', 'rb') as inputfile:
#    WordUctoTrainReviewsText = pickle.load(inputfile)
#    WordUctoTrainReviewsList = WordUctoTrainReviewsText.split("*$*$*$*$")

#print("loadeddddddddd")

#with open('total_review_morf_list_text.pickle', 'wb') as outputfile:
#  pickle.dump(total_review_morf_list_text, outputfile)

########## EXTRA EXPERIMENTING DATA ########

#with open('PosTestData.data', 'rb') as filehandle:
#    # read the data as binary data stream
#    PosTestCorpusList = pickle.load(filehandle)

#ShorterCorpusList = PosTestCorpusList[:100]
#short_text = ' '.join(ShorterCorpusList)
#pos_test_text = ' '.join(PosTestCorpusList)

###########

#print("CHECKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK")
#print(totTrainList[:10])

#print("WHATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
#print(MorphedTrainReviewsList[:10])
########## TRAIN TOKENIZER ##########

#all_text2 = ' '.join(totTrainList)
#morphSentences = change_text_to_morphs(totTestList)

print("LOADED")

totTrainList_Cleaned = []
for i in range(0,len(totTrainList)):
    cleaned_review = totTrainList[i].replace("\n"," ")
    totTrainList_Cleaned += [cleaned_review]

totTestList_Cleaned = []
for i in range(0,len(totTestList)):
    cleaned_review = totTestList[i].replace("\n"," ")
    totTestList_Cleaned += [cleaned_review]


#morfed_totTrainList = convertToPolyglotMorf(totTrainList_Cleaned, save=True)

#with open('convertedPolyglotMorf.pickle', 'rb') as inputfile:
#    morfed_totTrainList = pickle.load(inputfile)

#Open = open('covertedPolyglotMorfText.txt','r')
#morfed_totTrainList = Open.read()
#morfed_totTrainList = morfed_totTrainList.split('*%*%')
#print(morfed_totTrainList[:100])







#print(totTrainList_Cleaned[:3])
#print(totTrainList[:3])
#print(len(totTrainList))
#print(len(totTrainList_Cleaned))

#MorphedTrainReviewsText = ' '.join(MorphedTrainReviewsList)
#print(MorphedTrainReviewsText[:100])

print("LOADEDD")

#print("MORPHED")
#print("MORPHED")
#print("MORPHED")
#print("MORPHED")
#print("MORPHED")
#print(MorfedUctoTrainReviewsList[1:70])

#5000: 0.00041

encoder = Encoder(5000, pct_bpe=1, ngram_max=50)  ### params random still!
###Split by spaces (already ucto'd)
#encoder.fit(MorphedTrainReviewsList)
#encoder.fit(a_list_of_all_text2_morf_NLTK)
###Split by spaces (already ucto'd)
#encoder.fit(MorfedUctoTrainReviewsList)
###Split by spaces (already ucto'd)
#encoder.fit(SelfTrainedMorfedUctoTrainReviewsList)
###Split by spaces (already ucto'd) #CHANGE LINE IN MORPHISM ENCODER IF WORD TOKENIZATION ONLY
#encoder.fit(WordUctoTrainReviewsList)

### Polyglot Morf
encoder.fit(morfed_totTrainList)
### TotTrainList_cleaned
#encoder.fit(totTrainList_Cleaned)
### Lettergrepen
#encoder.fit(lettergreep_train_words)

print("finished fit")





########## CREATING FEATURES ##########

### Encode the train reviews (create features) ###

print("Encoding train wiki-words")
train_words_int = np.zeros((len(lettergreep_train_words), fixed_len), dtype = int)
i = 0

for word in lettergreep_train_words:
    train_word_int = encoder.transform(word, fixed_length = fixed_len)
    train_words_int[i,:] = np.array(train_word_int)
    if i % 100 == 0:
        print("wiki-word encoded: " + str(i))
    i += 1


### Encode the test reviews (create features) ###

print("Encoding test wiki-words")
test_words_int = np.zeros((len(lettergreep_test_words), fixed_len), dtype = int)
i = 0

for word in lettergreep_test_words:
    test_word_int = encoder.transform(word, fixed_length = fixed_len)
    test_words_int[i,:] = np.array(test_word_int)
    i += 1

### Adjust label lists to Arrays ###

train_labels = np.array(lettergreep_train_labels)
test_labels  = np.array(lettergreep_test_labels)
print(lettergreep_train_labels)
print(lettergreep_test_labels)


### Rename features ###

train_features = train_words_int
len_train_features = len(train_features)

test_features = test_words_int
len_test_features = len(test_features)

### Save features ###

np.save('LETTERGREEPTrainFeatures1000_5000.npy',train_features)
np.save('LETTERGREEPTestFeatures1000_5000.npy',test_features)
np.save('LETTERGREEPTrainLabels1000_5000.npy',train_labels)
np.save('LETTERGREEPTestLabels1000_5000.npy',test_labels)

print("okay")
print(encoder.bpe_vocab_size)
print(encoder.word_vocab_size)
print(len(encoder.bpe_vocab))
print(len(encoder.word_vocab))


print("BPE Vocab! ")
print(encoder.bpe_vocab)
print("")
print("")
print("word_vocab: ")
print(encoder.word_vocab)
print("")
print("HIGHEST VOCAB iNDEX IS")
print(max(list(encoder.bpe_vocab.values())))




#
#
randomEnglishSentence = "hey this is just a tokenized sentence do we need lowerlevel characters or different, who knows let's see if splelling mistakes ge interpreted"
nederlandseZin = "Hey heb je je aangetekende brief wel degelijk gekregen Ik hoop van wel"
nederlandseZin2 = "In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen."
nederlandseZin3 = "Dit is een korte nederlandse zin die getokenized moet worden"
nederlandseZin4 = "Wordt deze korte zin optimaal gesegmenteerd? zeker?? Ja hoor..."
test_twitterReview_neg = 'heeeeel sleeecht nie aan te zien zeer matig'

 #print("")
 #print(encoder.tokenize("dit is 雙喜 een zin die ik ga proberen tokenizen"))
 #print("")

tokenized = encoder.tokenize(nederlandseZin)
tokenized2 = encoder.tokenize(nederlandseZin2)
tokenized3 = encoder.tokenize(nederlandseZin3)
tokenized4 = encoder.tokenize(nederlandseZin4)
Ids = encoder.transform(nederlandseZin)
Ids2 = encoder.transform(nederlandseZin2)
Ids3 = encoder.transform(nederlandseZin3)
Ids4 = encoder.transform(nederlandseZin4)
#twitterIds = encoder.transform(test_twitterReview_neg)


tokens_without_eow = []
for token in tokenized:
 if not (token is encoder.EOW or token is encoder.SOW):
     tokens_without_eow += [token]

tokens_without_eow2 = []
for token in tokenized2:
 if not (token is encoder.EOW or token is encoder.SOW):
     tokens_without_eow2 += [token]

tokens_without_eow3 = []
for token in tokenized3:
 if not (token is encoder.EOW or token is encoder.SOW):
    tokens_without_eow3 += [token]

tokens_without_eow4 = []
for token in tokenized4:
 if not (token is encoder.EOW or token is encoder.SOW):
    tokens_without_eow4 += [token]

print("")
print("")
print("SENTENCE1")
print("")
print(nederlandseZin)
print("")
print("TOKENS")
print("")
print(tokens_without_eow)
print("")
print("ID's")
print("")
print(Ids)


print("")
print("")
print("SENTENCE2")
print("")
print(nederlandseZin2)
print("")
print("TOKENS")
print("")
print(tokens_without_eow2)
print("")
print("ID's")
print("")
print(Ids2)

print("")
print("")
print("SENTENCE3")
print("")
print(nederlandseZin3)
print("")
print("TOKENS")
print("")
print(tokens_without_eow3)
print("")
print("ID's")
print("")
print(Ids3)

print("")
print("")
print("SENTENCE4")
print("")
print(nederlandseZin4)
print("")
print("TOKENS")
print("")
print(tokens_without_eow4)
print("")
print("ID's")
print("")
print(Ids4)
print("")
print("")

print(format_time(time.time() - start_time))

####### CREATE MANUAL TEST REVIEWS #######

# fixed_len = 700
#       # test code and generate tokenized review
# test_review = 'heeeeel sleeecht, zeker nie leeswaardig'
# test_review2 = 'heeeeel sleeecht, nie aan te zien'
#
#        #Remove punctuation! (model trained without punctuation momentarily)
# test_review =''.join([c for c in test_review if c not in punctuation])
# features = np.zeros((1, fixed_len), dtype = int)
# review_int = encoder.transform(test_review, fixed_length = fixed_len)
# features[0,:] = np.array(review_int)
#
# test_review2 =''.join([c for c in test_review2 if c not in punctuation])
# features2 = np.zeros((1, fixed_len), dtype = int)
# review_int2 = encoder.transform(test_review2, fixed_length = fixed_len)
# features2[0,:] = np.array(review_int2)
#
# np.save('test_review1.npy',features)
# np.save('test_review2.npy',features2)

print(train_labels)
print(train_features)
max_len = 0
long_encoding = []
for i in train_features:
    length_nonzero = 0
    for j in i:
        if j != 0:
            length_nonzero += 1

    if length_nonzero == fixed_len:
        long_encoding += [i]

    if max_len < length_nonzero:
        max_len = length_nonzero

print(str(max_len) + " MAX TRAIN LENGTH")
print("Longest train encodings are: ")
print(long_encoding)

max_len = 0
long_encoding = []
for i in test_features:
    length_nonzero = 0
    for j in i:
        if j != 0:
            length_nonzero += 1
    if length_nonzero == fixed_len:
        long_encoding += [i]


    if max_len < length_nonzero:
        max_len = length_nonzero



print(str(max_len) + " MAX TEST LENGTH")
print("Longest test encodings are: ")
print(long_encoding)

#import IPython
#IPython ; IPython.embed() ; exit(1)