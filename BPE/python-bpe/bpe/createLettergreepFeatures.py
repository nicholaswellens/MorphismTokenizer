#from encoder import Encoder
import datetime
import time
from polyglot.text import Text, Word
import ucto
import os
import matplotlib.pyplot as plt
import IPython
import collections



import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from frog import Frog, FrogOptions
from numpy import asarray, savetxt, loadtxt
from string import punctuation
import glob

########################################


########## VARIABLES ##########

fixed_len = 20
split_frac = 0.9 #splitting fraction between TRAIN and TEST data

#from morphism_encoder import Encoder
from morphism_encoder import Encoder

### Nothing_Special
#from morphism_encoder_merge_lettergreep import Encoder

################ IMPORT LETTERGREPEN DATA #########################

# data = pd.read_csv("TrainFiles/lettergrepen_1.csv")
# data2 = pd.read_csv("TrainFiles/lettergrepen_2.csv")
# data3 = pd.read_csv("TrainFiles/lettergrepen_3.csv")
# data4 = pd.read_csv("TrainFiles/lettergrepen_4.csv")
# data5 = pd.read_csv("TrainFiles/lettergrepen_4,5.csv")
#
#
#
# print(data["woorden"])
# print(data["lettergrepen"])
# print(data["aantal lettergrepen"])
#
# lettergreep_words = list(data["woorden"])
# lettergreep_splits = list(data["lettergrepen"])
# lettergreep_labels = list(data["aantal lettergrepen"])
#
# lettergreep_words2 = list(data2["woorden"])
# lettergreep_splits2 = list(data2["lettergrepen"])
# lettergreep_labels2 = list(data2["aantal lettergrepen"])
#
# lettergreep_words3 = list(data3["woorden"])
# lettergreep_splits3 = list(data3["lettergrepen"])
# lettergreep_labels3 = list(data3["aantal lettergrepen"])
#
# lettergreep_words4 = list(data4["woorden"])
# lettergreep_splits4 = list(data4["lettergrepen"])
# lettergreep_labels4 = list(data4["aantal lettergrepen"])
#
# lettergreep_words5 = list(data5["woorden"])
# lettergreep_splits5 = list(data5["lettergrepen"])
# lettergreep_labels5 = list(data5["aantal lettergrepen"])
#
#
# lettergreep_words = lettergreep_words + lettergreep_words2 + lettergreep_words3 + lettergreep_words4 + lettergreep_words5
# lettergreep_splits = lettergreep_splits + lettergreep_splits2 + lettergreep_splits3 + lettergreep_splits4 + lettergreep_splits5
# lettergreep_labels = lettergreep_labels + lettergreep_labels2 + lettergreep_labels3 + lettergreep_labels4 + lettergreep_labels5

data = pd.read_csv("TrainFiles/lettergrepen_Official3.csv")
lettergreep_words = list(data["woorden"])
lettergreep_splits = list(data["lettergrepen"])
lettergreep_labels = list(data["aantal lettergrepen"])

len_dataset = len(lettergreep_words)

### Ree "#Name?" words..
for i in range(len_dataset):
    if i < len(lettergreep_words):
        if lettergreep_words[i] == "#NAME?":
            print("ok")
            lettergreep_words.pop(i)
            lettergreep_labels.pop(i)
            lettergreep_splits.pop(i)

len_dataset = len(lettergreep_words)
print(len_dataset)
print(len(lettergreep_splits))
print(len(lettergreep_labels))


lettergreep_train_words = lettergreep_words[0:int(split_frac*len_dataset)]
lettergreep_train_labels = lettergreep_labels[0:int(split_frac*len_dataset)]

lettergreep_test_words = lettergreep_words[int(split_frac*len_dataset):]
lettergreep_test_labels = lettergreep_labels[int(split_frac*len_dataset):]

#IPython ; IPython.embed() ; exit(1)

print("Number of train examples")
print(len(lettergreep_train_words))
print("")
print("Number of test examples")
print(len(lettergreep_test_words))
#########################################

print("Duplicates: ")
print([item for item, count in collections.Counter(lettergreep_words).items()if count >1])



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

## Taken from https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

print("TESTESSTTTT")
print(intersperse(["mee","speel","en","de"],"__ADD_MERGE__"))
print(intersperse(["ik"],"__ADD_MERGE__"))


def change_text_to_morphs(sentences, frog_merge = False,  save = False, filename=None):
    # sentence list to sentence list in frog morphism form
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
            if frog_merge:
                morphisms_word = list(filter(None, morphisms_word_list))
                morphisms_word_list = intersperse(morphisms_word_list, "insertmergetoken")
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
        morphSentences.append(morphSentence)

    if save is True:
        with open(filename, 'wb') as outputfile:
            pickle.dump(morphSentences, outputfile)
    return morphSentences

def change_text_to_lemma_POS(sentences,  save = False, filename=None):
    # sentence list to sentence list in frog lemma + pos
    lemmapos_sentences = []

    frog = Frog(FrogOptions(tok=True, lemma=True, morph=False, daringmorph=False, mwu=False, chunking=False, ner=False,
                            parser=False))

    for sentenceNumber in range(0, len(sentences)):
        print(sentenceNumber)
        print("of")
        print(len(sentences))
        sentenceToBeProcessed = sentences[sentenceNumber]
        #sentenceToBeProcessed = sentenceToBeProcessed.replace("\n", " ")

        output = frog.process(sentenceToBeProcessed)
        lemmapos_sentence = ""
        for i in range(0,len(output)):
            pos = str(output[i].get("pos"))
            lemma = str(output[i].get("lemma"))
            #posprob = str(output[i].get("posprob"))
            #print(posprob)

            # print("pos:      " + pos)
            # print("lemma:    " + lemma)

            pos = "<" + pos
            pos = pos.replace("(", "><")
            pos = pos.replace(")", ">")
            pos = pos.replace(",", "><")
            pos = pos.replace("<>", "")

            # print(pos)

            lemmapos_word = lemma + " " + pos

            #word = str(output[i].get("text"))
            #print(f"{word}: {lemmapos_word}")

            lemmapos_sentence = lemmapos_sentence + " " + lemmapos_word

        # Remove the first empty string
        #print(lemmapos_sentence)

        lemmapos_sentence = lemmapos_sentence[1:]
        #print("")
        #print("")
        #print("")
        #print("")
        #print(lemmapos_sentence)
        #print("")
        #print("")
        #print("")
        #print("")
        lemmapos_sentences.append(lemmapos_sentence)
        #print("")
        #print(lemmapos_sentences)
        #print("")

    if save is True:
        with open(filename, 'wb') as outputfile:
            pickle.dump(lemmapos_sentences, outputfile)
    return lemmapos_sentences

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
#    os.system("morfessor -t EenPosReview.txt -S model.segm -T EenPosReview.txt -o outputSelfMorf.txt ")

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

    morfed_sentences_text = '*%'.join(morfed_sentences)

    if save is True:

        with open("TrainFiles/convertedPolyglotMorfText.txt", "w") as text_file:
            text_file.write(morfed_sentences_text)

    return morfed_sentences



########## LOAD AND SHUFFLE DATA, CREATE LABELS ##########



### LOADING SHUFFLED TRAIN TEST DATA ###

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
print("LOADED")



### CLEANING TRAINLIST TESTLIST ###

totTrainList_Cleaned = []
for i in range(0,len(totTrainList)):
    cleaned_review = totTrainList[i].replace("\n"," ")
    totTrainList_Cleaned += [cleaned_review]

totTestList_Cleaned = []
for i in range(0,len(totTestList)):
    cleaned_review = totTestList[i].replace("\n"," ")
    totTestList_Cleaned += [cleaned_review]



### LOADING POLYGLOT MORFESSOR FILE ###

### Save new file
#morfed_totTrainList = convertToPolyglotMorf(totTrainList_Cleaned, save=True)

Open = open('TrainFiles/convertedPolyglotMorfText.txt','r')
morfed_totTrainList = Open.read()
morfed_totTrainList = morfed_totTrainList.split('*%')
print("-----------------------------")
print(morfed_totTrainList[:100])
print("-----------------------------")

### LOADING SELF-TRAINED MORFESSOR FILE ###

Open = open('TrainFiles/SelfMorfedTrainReviewsUnedited.txt','r')
SelfMorfed_totTrainListUnedited = Open.read()
SelfMorfed_totTrainListUnedited = SelfMorfed_totTrainListUnedited.split("\n")
SelfMorfed_totTrainList = []
for element in SelfMorfed_totTrainListUnedited:
    SelfMorfed_totTrainList += element.split(" ")

print("-----------------------------")
print(SelfMorfed_totTrainList[:100])
print("-----------------------------")



###########################################
### FROG FROG FROG IMPORT IMPORT IMPORT ###
###########################################

### LOADING FROG LEMMA POS TRAIN REVIEW FILE ###


#! PROBABLY NO NEED BECAUSE DONT HAVE SENTENCES FOR POS TAGS AT LETTERGREEP




#FrogLemmaPosTrainLettergreepSeperate = change_text_to_lemma_POS(lettergreep_train_words,  save = True, filename = 'TrainFiles/FrogLemmaPosTrainLettergreepSeperate.pickle')

#with open('TrainFiles/FrogLemmaPosTrainLettergreepSeperate.pickle', 'rb') as inputfile:
#    FrogLemmaPosTrainLettergreepSeperate = pickle.load(inputfile)

with open('TrainFiles/FrogLemmaPosTrainReviewsSeperate.pickle', 'rb') as inputfile:
    FrogLemmaPosTrainReviewsSeperate = pickle.load(inputfile)


# print("Before First")
# print("")
# print(FrogLemmaPosTrainLettergreepSeperate[:10])

print("First")
print("")
print(FrogLemmaPosTrainReviewsSeperate[:10])

#FrogLemmaPosTestLettergreepSeperate = change_text_to_lemma_POS(totTestList_Cleaned,  save = True, filename = 'TrainFiles/FrogLemmaPosTestLettergreepSeperate.pickle')

#with open('TrainFiles/FrogLemmaPosTestLettergreepSeperate.pickle', 'rb') as inputfile:
#    FrogLemmaPosTestLettergreepSeperate = pickle.load(inputfile)

# print("Second")
# print("")
# print(FrogLemmaPosTestLettergreepSeperate[:5])







### LOADING FROG MORPHISM TRAIN REVIEW FILE ###

#FrogMorphed_totTrainList = change_text_to_morphs(totTrainList_Cleaned,  save = True)
with open('TrainFiles/FrogMorphedTrainReviews.pickle', 'rb') as inputfile:
    FrogMorphedTrainReviews = pickle.load(inputfile)

#FrogMorphed_totTestList = change_text_to_morphs(totTestList_Cleaned,  save = True, filename = 'TrainFiles/FrogMorphedTestReviews.pickle')
with open('TrainFiles/FrogMorphedTestReviews.pickle', 'rb') as inputfile:
    FrogMorphedTestReviews = pickle.load(inputfile)

print("-----------------------------")
print(FrogMorphedTrainReviews[:100])
print("-----------------------------")




### LOADING FROG MORPHISM TRAIN TEST LETTERGREEP FILES ###

#FrogMorphedTrainLettergreep = change_text_to_morphs(lettergreep_train_words,  save = True, filename = 'TrainFiles/FrogMorphedTrainLettergreep.pickle')
with open('TrainFiles/FrogMorphedTrainLettergreep.pickle', 'rb') as inputfile:
    FrogMorphedTrainLettergreep = pickle.load(inputfile)

#FrogMorphedTestLettergreep = change_text_to_morphs(lettergreep_test_words,  save = True, filename = 'TrainFiles/FrogMorphedTestLettergreep.pickle')
with open('TrainFiles/FrogMorphedTestLettergreep.pickle', 'rb') as inputfile:
    FrogMorphedTestLettergreep = pickle.load(inputfile)


print("-----------------------------")
#print(FrogMorphed_totTrainList[:100])
print("-----------------------------")


print("LOADEDD")


#5000: 0.00041

encoder = Encoder(30000, pct_bpe=0.001, ngram_max=50)  ### params random still!
###Split by spaces (already ucto'd)
#encoder.fit(MorphedTrainReviewsList)
#encoder.fit(a_list_of_all_text2_morf_NLTK)
###Split by spaces (already ucto'd)
#encoder.fit(MorfedUctoTrainReviewsList)
###Split by spaces (already ucto'd)
#encoder.fit(SelfTrainedMorfedUctoTrainReviewsList)
###Split by spaces (already ucto'd) #CHANGE LINE IN MORPHISM ENCODER IF WORD TOKENIZATION ONLY
#encoder.fit(WordUctoTrainReviewsList)


### Frog Morph                                   Whitespace..Whitespace
#encoder.fit(FrogMorphedTrainReviews)
### Polyglot Morf                                Ucto..Whitespace
#encoder.fit(morfed_totTrainList)
### Selftrained Morf                             Ucto..Whitespace
#encoder.fit(SelfMorfed_totTrainList)
### TotTrainList_cleaned FOR BPE AND WORD        Ucto..Ucto          WORD: update file
encoder.fit(totTrainList_Cleaned)



### Lettergrepen                                 Ucto..Ucto
#encoder.fit(lettergreep_train_words)

print("finished fit")





########## CREATING FEATURES ##########

### Encode the train reviews (create features) ###

###For length purposes only:
for_length_words_int = []

print("Encoding train wiki-words")

#train_words_int = np.zeros((len(FrogMorphedTrainLettergreep), fixed_len), dtype = int)
train_words_int = np.zeros((len(lettergreep_train_words), fixed_len), dtype = int)
i = 0

for word in lettergreep_train_words:
    train_word_int, train_word_int_without_padding = encoder.transform(word, fixed_length = fixed_len)
    train_words_int[i,:] = np.array(train_word_int)
    if i % 100 == 0:
        print("wiki-word encoded: " + str(i))
    i += 1
    ###For length purposes only:
    for_length_words_int.append(train_word_int_without_padding)

### Encode the test reviews (create features) ###

print("Encoding test wiki-words")
#test_words_int = np.zeros((len(FrogMorphedTestLettergreep), fixed_len), dtype = int)
test_words_int = np.zeros((len(lettergreep_test_words), fixed_len), dtype = int)
i = 0

for word in lettergreep_test_words:
    test_word_int, test_word_int_without_padding = encoder.transform(word, fixed_length = fixed_len)
    test_words_int[i,:] = np.array(test_word_int)
    i += 1
    ###For length purposes only:
    for_length_words_int.append(test_word_int_without_padding)

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

np.save('LETTERGREEP_NoMERGE_WORD_TrainFeatures_30000.npy',train_features)
np.save('LETTERGREEP_NoMERGE_WORD_TestFeatures_30000.npy',test_features)
np.save('LETTERGREEP_NoMERGE_WORD_TrainLabels_30000.npy',train_labels)
np.save('LETTERGREEP_NoMERGE_WORD_TestLabels_30000.npy',test_labels)

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


############## SHOWING SEQUENCE LENGTHS ########################

print("Calculating train sequence lengths")

### Calculate and show lengths of reviews...

words_len = [len(x) for x in for_length_words_int]
pd.Series(words_len).hist()
pd.Series(words_len).describe().to_csv("InfoDatasetLettergreep/30000-NoMerge-Word.csv")
print(pd.Series(words_len).describe())
plt.xlabel = "Review Length"
plt.ylabel = "Amount of reviews"
#plt.legend()
plt.savefig("InfoDatasetLettergreep/30000-NoMerge-Word.png",format='png')





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
Ids,_ = encoder.transform(nederlandseZin)
Ids2,_ = encoder.transform(nederlandseZin2)
Ids3,_ = encoder.transform(nederlandseZin3)
Ids4,_ = encoder.transform(nederlandseZin4)
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

for word in lettergreep_test_words[:3]:
    print(encoder.tokenize(word))
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