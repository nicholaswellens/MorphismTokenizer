#from encoder import Encoder
import datetime
import time



import pickle
import numpy as np
from frog import Frog, FrogOptions
import ucto
from polyglot.text import Text, Word
from numpy import asarray, savetxt, loadtxt
from string import punctuation
#import tkinter
#import _tkinter
import pandas as pd
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


########## VARIABLES ##########

fixed_len = 2500
from morphism_encoder_hashtags import Encoder
#from morphism_encoder_merge import Encoder

Morphing = False

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


#Deleting of \n in reviews, maybe can use for sentence tag putting.
#Misschien wel goed zo, maar als je vergelijkt met word-tokenization misschien ook verwijderen in de google colab

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
                morphisms_word_list = list(filter(None, morphisms_word_list))
                morphisms_word_list = intersperse(morphisms_word_list, "insertmergetoken")
                #print(morphisms_word_list)
            #print("EVET")
            #print(morphisms_word_list)
            morphSentence += morphisms_word_list
        #print("MORPHSENTENCE")
        #print(morphSentence)
        # Remove the empty strings
        morphSentence = list(filter(None, morphSentence))
        #print("ok")
        #print(morphSentence)
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
        sentenceToBeProcessed = sentenceToBeProcessed.replace("\n", " ")

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


### LOADING SHUFFLED TRAIN TEST DATA ###

print("loadedStart")

# with open('totTrainListShuffled.data', 'rb') as filehandle:
#       # read the data as binary data stream
#       totTrainList = pickle.load(filehandle)

#print("loaded")

with open('trainLabelListShuffled.data', 'rb') as filehandle:
      # read the data as binary data stream
      trainLabelList = pickle.load(filehandle)

print("loadedd")

with open('totTestListShuffled.data', 'rb') as filehandle:
      # read the data as binary data stream
      totTestList = pickle.load(filehandle)

# with open('totTrainListShuffled.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     totTrainList = pickle.load(filehandle)
totTrainList = totTestList
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
#print(morfed_totTrainList[:100])
print("-----------------------------")

### LOADING SELF-TRAINED MORFESSOR FILE ###

Open = open('TrainFiles/SelfMorfedTrainReviewsUnedited.txt','r')
SelfMorfed_totTrainListUnedited = Open.read()
SelfMorfed_totTrainListUnedited = SelfMorfed_totTrainListUnedited.split("\n")
SelfMorfed_totTrainList = []
for element in SelfMorfed_totTrainListUnedited:
    SelfMorfed_totTrainList += element.split(" ")

print("-----------------------------")
#print(SelfMorfed_totTrainList[:100])
print("-----------------------------")



###########################################
### FROG FROG FROG IMPORT IMPORT IMPORT ###
###########################################

### LOADING FROG LEMMA POS TRAIN REVIEW FILE ###

#FrogLemmaPosTrainReviews = change_text_to_lemma_POS(totTrainList_Cleaned,  save = True, filename = 'TrainFiles/FrogLemmaPosTrainReviewsSeperate.pickle')
with open('TrainFiles/FrogLemmaPosTrainReviews.pickle', 'rb') as inputfile:
    FrogLemmaPosTrainReviews = pickle.load(inputfile)

with open('TrainFiles/FrogLemmaPosTrainReviewsSeperate.pickle', 'rb') as inputfile:
    FrogLemmaPosTrainReviewsSeperate = pickle.load(inputfile)

print("First")
print("")
#print(FrogLemmaPosTrainReviews[:10])

#FrogLemmaPosTestReviews = change_text_to_lemma_POS(totTestList_Cleaned,  save = True, filename = 'TrainFiles/FrogLemmaPosTestReviewsSeperate.pickle')
with open('TrainFiles/FrogLemmaPosTestReviews.pickle', 'rb') as inputfile:
    FrogLemmaPosTestReviews = pickle.load(inputfile)

with open('TrainFiles/FrogLemmaPosTestReviewsSeperate.pickle', 'rb') as inputfile:
    FrogLemmaPosTestReviewsSeperate = pickle.load(inputfile)

print("Second")
print("")
#print(FrogLemmaPosTestReviews[:5])


### LOADING FROG MORPHISM TRAIN REVIEW FILE ###


#FrogMorphedTrainReviews = change_text_to_morphs(totTrainList_Cleaned,  save = True, filename = 'TrainFiles/FrogMorphedTrainReviews.pickle')
with open('TrainFiles/FrogMorphedTrainReviews.pickle', 'rb') as inputfile:
    FrogMorphedTrainReviews = pickle.load(inputfile)

#FrogMorphedTestReviews = change_text_to_morphs(totTestList_Cleaned,  save = True, filename = 'TrainFiles/FrogMorphedTestReviews.pickle')
with open('TrainFiles/FrogMorphedTestReviews.pickle', 'rb') as inputfile:
    FrogMorphedTestReviews = pickle.load(inputfile)


### LOADING FROG MORPHISM TRAIN REVIEW FILE FOR TRANSFORMING MERGE PURPOSES... SPECIAL FUNCTION IN ENCODER... ###

#FrogMorphedTrainReviewsMergeEdited = change_text_to_morphs(totTrainList_Cleaned, frog_merge = True, save = True, filename = 'TrainFiles/FrogMorphedTrainReviewsMergeEdited.pickle')
with open('TrainFiles/FrogMorphedTrainReviewsMergeEdited.pickle', 'rb') as inputfile:
    FrogMorphedTrainReviewsMergeEdited = pickle.load(inputfile)
#    for i in range(len(FrogMorphedTrainReviewsMergeEdited)):
#        FrogMorphedTrainReviewsMergeEdited[i] = FrogMorphedTrainReviewsMergeEdited[i].replace("__ADD_MERGE__","insertmergetoken")
#        print(i)
#FrogMorphedTestReviewsMergeEdited = change_text_to_morphs(totTestList_Cleaned, frog_merge = True,  save = True, filename = 'TrainFiles/FrogMorphedTestReviewsMergeEdited.pickle')
with open('TrainFiles/FrogMorphedTestReviewsMergeEdited.pickle', 'rb') as inputfile:
    FrogMorphedTestReviewsMergeEdited = pickle.load(inputfile)
#    for i in range(len(FrogMorphedTestReviewsMergeEdited)):
#        FrogMorphedTestReviewsMergeEdited[i] = FrogMorphedTestReviewsMergeEdited[i].replace("__ADD_MERGE__","insertmergetoken")
#        print(i)

print("-----------------------------")
#print(FrogMorphedTrainReviews[:5])
print("-----------------------------")
print("-----------------------------")
print("-----------------------------")
print("-----------------------------")
#print(FrogMorphedTrainReviewsMergeEdited[:5])
print("-----------------------------")

#2000: 0.001
#5000: 0.00041
#30000: 0.00006666666

encoder = Encoder(30000, pct_bpe=1, ngram_max=50)  ### params random still!
###Split by spaces (already ucto'd)
#encoder.fit(MorphedTrainReviewsList)
#encoder.fit(a_list_of_all_text2_morf_NLTK)
###Split by spaces (already ucto'd)
#encoder.fit(MorfedUctoTrainReviewsList)
###Split by spaces (already ucto'd)
#encoder.fit(SelfTrainedMorfedUctoTrainReviewsList)
#encoder.fit(SelfTrainedMorfedUctoTrainReviewsListSplit)
###Split by spaces (already ucto'd) #CHANGE LINE IN MORPHISM ENCODER IF WORD TOKENIZATION ONLY
#encoder.fit(WordUctoTrainReviewsList)
###SAME IF UCTO IS ON###
#encoder.fit(totTrainList_Cleaned)

nederlandseZin4 = "Wordt deze korte nederlandse zin optimaal gesegmenteerd"
nederlandseZin3 = "Dit is een korte nederlandse zin die getokenized moet worden"
#### IF MORPH ####
if Morphing:
    #[nederlandseZin] = change_text_to_morphs([nederlandseZin])
    #[nederlandseZin2] = change_text_to_morphs([nederlandseZin2])
    #[nederlandseZin33] = change_text_to_morphs([nederlandseZin3])
    #[nederlandseZin44] = change_text_to_morphs([nederlandseZin4])
    #[randomEnglishSentence] = change_text_to_morphs([randomEnglishSentence])

    #[nederlandseZin222] = change_text_to_morphs([nederlandseZin2])
    #[nederlandseZin333] = change_text_to_morphs([nederlandseZin3])
    nederlandseZin444 = change_text_to_lemma_POS([nederlandseZin4])[0]
    nederlandseZin333 = change_text_to_lemma_POS([nederlandseZin3])[0]
    print("example sentences converted")
    #print(nederlandseZin333)
    #print(nederlandseZin444)
    #print(nederlandseZin333)


print("started fit")
#print(len(FrogLemmaPosTrainReviews))

### Frog Morph                                   Whitespace..Whitespace
print("DOWN")
print(FrogMorphedTrainReviews[0:10])
print("UP")
encoder.fit(FrogMorphedTrainReviews)


### Polyglot Morf                                Ucto..Whitespace
#encoder.fit(morfed_totTrainList)
### Selftrained Morf                             Ucto..Whitespace
#encoder.fit(SelfMorfed_totTrainList)
### TotTrainList_cleaned FOR BPE AND WORD        Ucto..Ucto          WORD: update file
#encoder.fit(totTrainList_Cleaned)

### Frog LemmaPos                                Whitespace..Whitespace
#encoder.fit(FrogLemmaPosTrainReviews)

### Frog LemmaPos Seperate                       Whitespace..Whitespace
##print("DOWN")
##print(FrogLemmaPosTrainReviewsSeperate[0:10])
##print("UP")
#encoder.fit(FrogLemmaPosTrainReviewsSeperate)

print("finished fit")



#######################################################

########## CREATING FEATURES ##########

### Encode the train reviews (create features) ###


###For length purposes only:
for_length_reviews_int = []

print("Encoding train reviews")


#train_reviews_int = np.zeros((len(FrogLemmaPosTrainReviews), fixed_len), dtype = int)
#train_reviews_int = np.zeros((len(FrogLemmaPosTrainReviewsSeperate), fixed_len), dtype = int)

### need "frog_merge = True" in transform function if MERGE token for FROG
#train_reviews_int = np.zeros((len(FrogMorphedTrainReviewsMergeEdited), fixed_len), dtype = int)
#train_reviews_int = np.zeros((len(FrogMorphedTrainReviews), fixed_len), dtype = int)

train_reviews_int = np.zeros((len(totTrainList_Cleaned), fixed_len), dtype = int)

i = 0

for review in totTrainList_Cleaned:
    train_review_int, train_review_int_without_padding = encoder.transform(review, fixed_length = fixed_len)
    train_reviews_int[i,:] = np.array(train_review_int)
    if i % 100 == 0:
        print("review encoded: " + str(i))
    i += 1
    ###For length purposes only:
    for_length_reviews_int.append(train_review_int_without_padding)


### Encode the test reviews (create features) ###

print("Encoding test reviews")

#test_reviews_int = np.zeros((len(FrogLemmaPosTestReviews), fixed_len), dtype = int)
#test_reviews_int = np.zeros((len(FrogLemmaPosTestReviewsSeperate), fixed_len), dtype = int)

### need "frog_merge = True" in transform function if MERGE token for FROG
#test_reviews_int = np.zeros((len(FrogMorphedTestReviewsMergeEdited), fixed_len), dtype = int)
#test_reviews_int = np.zeros((len(FrogMorphedTestReviews), fixed_len), dtype = int)

test_reviews_int = np.zeros((len(totTestList_Cleaned), fixed_len), dtype = int)

i = 0

for review in totTestList_Cleaned:
    test_review_int, test_review_int_without_padding = encoder.transform(review, fixed_length = fixed_len)
    test_reviews_int[i,:] = np.array(test_review_int)
    i += 1
    ###For length purposes only:
    for_length_reviews_int.append(test_review_int_without_padding)

### Adjust label lists to Arrays ###

train_labels = np.array(trainLabelList)
test_labels  = np.array(testLabelList)
print(train_labels)
print(test_labels)

### Rename features ###

train_features = train_reviews_int
len_train_features = len(train_features)

test_features = test_reviews_int
len_test_features = len(test_features)

### Save features ###

np.save('NoMERGE_BPE_TrainFeatures_30000_2500SeqLen.npy',train_features)
np.save('NoMERGE_BPE_TestFeatures_30000_2500SeqLen.npy',test_features)
np.save('NoMERGE_BPE_TrainLabels_30000_2500SeqLen.npy',train_labels)
np.save('NoMERGE_BPE_TestLabels_30000_2500SeqLen.npy',test_labels)

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

### Analyze Train Reviews Length WITHOUT EOW/SOW ###

#!!
#test_reviews_int = np.zeros((len(MorphedTestReviewsList), fixed_len), dtype = int)


#
# train_reviews_int = []
# for review in totTrainList_Cleaned:
#     train_review_int = encoder.transform(review)
#     train_reviews_int.append(train_review_int)
#
# train_reviews_len = [len(x) for x in train_reviews_int]
# pd.Series(train_reviews_len).hist()
# pd.Series(train_reviews_len).describe().to_csv("InfoDataset/Train-30000-NoMerge-Polyglot-Morf.csv")
# print(pd.Series(train_reviews_len).describe())
# plt.xlabel = "Review Length"
# plt.ylabel = "Amount of reviews"
# #plt.legend()
# plt.savefig("InfoDataset/Train-30000-NoMerge-Polyglot-Morf.png",format='png')
#
#
#
# ### Analyze Test Reviews Length WITHOUT EOW/SOW ###
#
# print("Calculating test sequence lengths")
#
# test_reviews_int = []
# for review in totTestList_Cleaned:
#     test_review_int = encoder.transform(review)
#     test_reviews_int.append(test_review_int)
#
#
# test_reviews_len = [len(x) for x in test_reviews_int]
# pd.Series(test_reviews_len).hist()
# pd.Series(test_reviews_len).describe().to_csv("InfoDataset/Test-30000-NoMerge-Polyglot-Morf.csv")
# print(pd.Series(test_reviews_len).describe())
# plt.xlabel = "Review Length"
# plt.ylabel = "Amount of reviews"
# #plt.legend()
# plt.savefig("InfoDataset/Test-30000-NoMerge-Polyglot-Morf.png",format='png')
#
# Analyze Train Reviews Length


#!!
#test_reviews_int = np.zeros((len(MorphedTestReviewsList), fixed_len), dtype = int)

# reviews_int = []
# for review in totTrainList_Cleaned:
#     review_int = encoder.transform(review)
#     reviews_int.append(review_int)
# test_reviews_int = []
# for review in totTestList_Cleaned:
#     review_int = encoder.transform(review)
#     reviews_int.append(review_int)
#
#
# reviews_len = [len(x) for x in reviews_int]
# pd.Series(reviews_len).hist()
# pd.Series(reviews_len).describe().to_csv("InfoDataset/30000-NoMerge-Polyglot-Morf.csv")
# print(pd.Series(reviews_len).describe())
# plt.xlabel = "Review Length"
# plt.ylabel = "Amount of reviews"
# #plt.legend()
# plt.savefig("InfoDataset/30000-NoMerge-Polyglot-Morf.png",format='png')
#







# # # reviews_len = [len(x) for x in for_length_reviews_int]
# # # pd.Series(reviews_len).hist()
# # # pd.Series(reviews_len).describe().to_csv("InfoDataset/30000-NoMerge-LemmaPosSep.csv")
# # # print(pd.Series(reviews_len).describe())
# # # plt.xlabel = "Review Length"
# # # plt.ylabel = "Amount of reviews"
# # # #plt.legend()
# # # plt.savefig("InfoDataset/30000-NoMerge-LemmaPosSep.png",format='png')






# from pandas.plotting import table
#
# reviews_len = [len(x) for x in for_length_reviews_int]
# pd.Series(reviews_len).hist()
# desc = pd.Series(reviews_len).describe()
# plot = plt.subplot(111,frame_on = False)
# plot.xaxis.set_visible(False)
# plot.yaxis.set_visible(False)
# table(plot, desc, loc ='upper right')
# plt.savefig("InfoDataset/30000-NooooMerge-Polyglot-Morf.png",format='png')


########## TESTING AND PRINTING ##################




# ############################################
# #
# randomEnglishSentence = "hey this is just a tokenized sentence do we need lowerlevel characters or different, who knows let's see if splelling mistakes ge interpreted"
# nederlandseZin = "Hey heb je je aangetekende brief wel degelijk gekregen Ik hoop van wel"
# nederlandseZin2 = "In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen."
# nederlandseZin3 = "Dit is een korte nederlandse zin die getokenized moet worden"
# #nederlandseZin4 = "Wordt deze korte zin optimaal gesegmenteerd? zeker?? Ja hoor..."
# nederlandseZin4 = "Wordt deze korte nederlandse zin optimaal gesegmenteerd"
# test_twitterReview_neg = 'heeeeel sleeecht nie aan te zien zeer matig'
#
#
#
#
#  #print("")
#  #print(encoder.tokenize("dit is 雙喜 een zin die ik ga proberen tokenizen"))
#  #print("")
#
# tokenized = encoder.tokenize(nederlandseZin)
# tokenized2 = encoder.tokenize(nederlandseZin2)
# tokenized3 = encoder.tokenize(nederlandseZin333)
# tokenized4 = encoder.tokenize(nederlandseZin444)
# Ids,_ = encoder.transform(nederlandseZin)
# Ids2,_ = encoder.transform(nederlandseZin2)
# Ids3,_ = encoder.transform(nederlandseZin333)
# #Ids4,_ = encoder.transform(nederlandseZin4)
# Ids4,_ = encoder.transform(nederlandseZin444)
# #twitterIds = encoder.transform(test_twitterReview_neg)
#
#
# tokens_without_eow = []
# for token in tokenized:
#  if not (token is encoder.EOW or token is encoder.SOW):
#      tokens_without_eow += [token]
#
# tokens_without_eow2 = []
# for token in tokenized2:
#  if not (token is encoder.EOW or token is encoder.SOW):
#      tokens_without_eow2 += [token]
#
# tokens_without_eow3 = []
# for token in tokenized3:
#  if not (token is encoder.EOW or token is encoder.SOW):
#     tokens_without_eow3 += [token]
#
# tokens_without_eow4 = []
# for token in tokenized4:
#  if not (token is encoder.EOW or token is encoder.SOW):
#     tokens_without_eow4 += [token]
#
# print("")
# print("")
# print("SENTENCE1")
# print("")
# print(nederlandseZin)
# print("")
# print("TOKENS")
# print("")
# print(tokens_without_eow)
# print("")
# print("ID's")
# print("")
# print(Ids)
#
#
# print("")
# print("")
# print("SENTENCE2")
# print("")
# print(nederlandseZin2)
# print("")
# print("TOKENS")
# print("")
# print(tokens_without_eow2)
# print("")
# print("ID's")
# print("")
# print(Ids2)
#
# print("")
# print("")
# print("SENTENCE3")
# print("")
# print(nederlandseZin3)
# print(nederlandseZin333)
# print("")
# print("TOKENS")
# print("")
# print(tokens_without_eow3)
# print("")
# print("ID's")
# print("")
# print(Ids3)
#
# print("")
# print("")
# print("SENTENCE4")
# print("")
# print(nederlandseZin4)
# print(nederlandseZin444)
# print("")
# print("TOKENS")
# print("")
# print(tokens_without_eow4)
# print("")
# print("ID's")
# print("")
# print(Ids4)
# print("")
# print("")
#
# print(format_time(time.time() - start_time))
#
# ####### CREATE MANUAL TEST REVIEWS #######
#
# # fixed_len = 700
# #       # test code and generate tokenized review
# # test_review = 'heeeeel sleeecht, zeker nie leeswaardig'
# # test_review2 = 'heeeeel sleeecht, nie aan te zien'
# #
# #        #Remove punctuation! (model trained without punctuation momentarily)
# # test_review =''.join([c for c in test_review if c not in punctuation])
# # features = np.zeros((1, fixed_len), dtype = int)
# # review_int = encoder.transform(test_review, fixed_length = fixed_len)
# # features[0,:] = np.array(review_int)
# #
# # test_review2 =''.join([c for c in test_review2 if c not in punctuation])
# # features2 = np.zeros((1, fixed_len), dtype = int)
# # review_int2 = encoder.transform(test_review2, fixed_length = fixed_len)
# # features2[0,:] = np.array(review_int2)
# #
# # np.save('test_review1.npy',features)
# # np.save('test_review2.npy',features2)
#
#

