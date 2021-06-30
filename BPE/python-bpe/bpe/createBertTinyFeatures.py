import datetime
import time

import pickle
import numpy as np
from frog import Frog, FrogOptions
import ucto
from polyglot.text import Text, Word
from numpy import asarray, savetxt, loadtxt
from string import punctuation
import pandas as pd
import matplotlib.pyplot as plt


# ########## VARIABLES ##########

fixed_len = 128
from BERTTINY_morphism_encoder_LemmaPos import Encoder


# ########## HELPER FUNCTIONS ############
#
start_time = time.time()

def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''
  # Round to the nearest second.
  elapsed_rounded = int(round((elapsed)))

  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))
#
#
#
## Taken from https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

print("TESTESSTTTT")
print(intersperse(["mee","speel","en","de"],"__ADD_MERGE__"))
print(intersperse(["ik"],"__ADD_MERGE__"))
#
#
# #Deleting of \n in reviews, maybe can use for sentence tag putting.
# #Misschien wel goed zo, maar als je vergelijkt met word-tokenization misschien ook verwijderen in de google colab
#
def change_text_to_morphs(sentences, frog_merge = False,  save = False, filename=None):
    # sentence list to sentence list in frog morphism form
    morphSentences = []

    frog = Frog(
        FrogOptions(tok=True, lemma=True, morph=True, daringmorph=False, mwu=False, chunking=False, ner=False,
                    parser=False))
    j = 0
    for sentenceToBeProcessed in sentences:

        if j % 1000 == 0:
            print(j+1)
            print("of")
            print(len(sentences))
        
        j += 1
        sentenceToBeProcessed = sentenceToBeProcessed.rstrip('\n')
        morphSentence = []
        output = frog.process(sentenceToBeProcessed)

        for i in range(0,len(output)):
            morphisms_word = output[i].get("morph")
            morphisms_word_list = morphisms_word.replace('[', '').split(']')
            if frog_merge:
                morphisms_word_list = list(filter(None, morphisms_word_list))
                morphisms_word_list = intersperse(morphisms_word_list, "__add_merge__")

            morphSentence += morphisms_word_list

        # Remove the empty strings
        morphSentence = list(filter(None, morphSentence))

        morphSentence = ' '.join(morphSentence)
        morphSentences.append(morphSentence)

    if save is True:
        with open(filename, 'wb') as outputfile:
            pickle.dump(morphSentences, outputfile)
    return morphSentences


# #2000: 0.001
# #5000: 0.00041
# #30000: 0.00006666666

def change_text_to_lemma_POS(sentences, save = False, filename=None):
    # sentence list to sentence list in frog lemma + pos
    lemmapos_sentences = []

    frog = Frog(FrogOptions(tok=True, lemma=True, morph=False, daringmorph=False, mwu=False, chunking=False, ner=False,
                            parser=False))

    j =0
    for sentenceToBeProcessed in sentences:
        if j % 1000 == 0:
            print(j + 1)
            print("of")
            print(len(sentences))

        j += 1
        sentenceToBeProcessed = sentenceToBeProcessed.rstrip('\n')
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

            lemmapos_word = lemma + " " + "**" + pos + "**"

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






loading_files_start_time = time.time()



### LOADING FROG LEMMA POS TRAIN REVIEW FILE ###

# def import_europarl_dataset(path):
#     dataset = pd.read_csv(path, 'utf-8', header=None, names=['Dutch'], engine='python')
#     dataset = dataset.to_numpy()
#     dataset = [item for sublist in dataset for item in sublist]
#     return dataset
#
# full_dataset = import_europarl_dataset('TrainFiles-BERT-TINY/europarl-v7.nl-en.nl')
# europarl_dataset = full_dataset[20000:]
# print("Sample of the europarl dataset: ")
# print(europarl_dataset[0:5])
# print("amount of europarl sentences: ")
# print(len(europarl_dataset))
#
# europarl_dataset_1400k_2000k = europarl_dataset[1400000:2000000]
# this_time = time.time()
# Tester = change_text_to_lemma_POS(europarl_dataset_1400k_2000k,  save = True, filename = 'TrainFiles-BERT-TINY/FrogLemmaPosEuroparl1400k_2000k.pickle')
# print(format_time(time.time() - this_time))
# print("this is time of 1400k_2000k sentences lemma POS process")
# print(Tester[:10])


# with open("TrainFiles-BERT-TINY/nl_part_1.txt") as f:
#     lines = f.readlines()
#
# oscar_dataset_newlines = lines[:2000000]
# #oscar_dataset = oscar_dataset_newlines
# oscar_dataset_1700k_2000k = oscar_dataset_newlines[1700000:]
# this_time = time.time()
# Tester = change_text_to_lemma_POS(oscar_dataset_1700k_2000k,  save = True, filename = 'TrainFiles-BERT-TINY/FrogLemmaPosOscar1700k_2000k.pickle')
# print(format_time(time.time() - this_time))
# print("this is time of 1700k_2000k sentences lemma POS process FOR OSCAR")
# print(Tester[:2])

#4,5 or 5 hours
#this is time of 700k sentences lemma POS process

#2:49:52
#this is time of 700k_1400k sentences lemma POS process

#1:43:23
#this is time of 1400k_2000k sentences lemma POS process

#8:42:01
#this is time of 100k_400k sentences lemma POS process FOR OSCAR

#8:33:45
#this is time of 400k_600k sentences lemma POS process FOR OSCAR

#11:28:27
#this is time of 600k_1000k sentences lemma POS process FOR OSCAR

#12:33:02
#this is time of 1000k_1300k sentences lemma POS process FOR OSCAR

#3:24:31
#this is time of 1300k_1400k sentences lemma POS process FOR OSCAR

#11:40:00
#this is time of 1400k_1700k sentences lemma POS process FOR OSCAR

#10:36:08
#this is time of 1700k_2000k sentences lemma POS process FOR OSCAR




########## BPE hastags (because ucto)
#1,5 million europarl:
#Total fit time:
#2:57:22
#creating_and_fitting_time is: 2:58:00

#       30521
#   ##: 22690
#   normal: 7827

#500k oscar encode: 1:47:00
#OSCAR ENCODING TIME 500k-1000k: 
#1:22:50

#OSCAR ENCODING TIME: 1000k-2000k
#4:34:05

##Europarl: 1000k
#EUROPARL ENCODING TIME:
#0:53:31

#EUROPARL ENCODING TIME:
#1:01:14


######## LEMMAPOS

#OSCAR ENCODING TIME 500k
#0:22:58


#OSCAR ENCODING TIME: 500-1000 
#0:28:39

#OSCAR ENCODING TIME: 1000-2000     EUROPARL ENCODING TIME: 
                                    #0:16:10

#0:29:36


#EUROPARL ENCODING TIME: 1000k  EUROPARL ENCODING TIME: 
                                #0:14:45

#0:17:20

###########################PRETRAIN TIME AND INFO#########

# MORPHMERGE: 100%|██████████| 9439160/9439160 [77:46:18<00:00, 33.71it/s]
#{'eval_loss': 1.0451945066452026, 'eval_runtime': 86.2608, 'eval_samples_per_second': 2303.711, 'epoch': 40.0}
#{'eval_loss': 1.048019528388977, 'eval_runtime': 86.9014, 'eval_samples_per_second': 2286.729, 'epoch': 39.0}
#{'eval_loss': 1.0511647462844849, 'eval_runtime': 86.354, 'eval_samples_per_second': 2301.226, 'epoch': 38.0}








### LOADING SELF-TRAINED MORFESSOR FILE FOR BPE ###

#with open('TrainFiles-BERT-TINY/morfessorOutput1000k1000k.txt','r') as Open:
#    SelfMorfed_totTrainListUnedited = Open.read()
#SelfMorfed_totTrainListUnedited = SelfMorfed_totTrainListUnedited.split("\n")
#BERTTINY_MORFESSOR_TRAIN_TOKENIZER = []
#for element in SelfMorfed_totTrainListUnedited:
#    BERTTINY_MORFESSOR_TRAIN_TOKENIZER += element.split(" ")

#print("-----------------------------")
#print(BERTTINY_MORFESSOR_TRAIN_TOKENIZER[:1000])
#print("-----------------------------")

#print("size of tokenizer training data")
#print(len(BERTTINY_MORFESSOR_TRAIN_TOKENIZER))

















# def import_europarl_dataset(path):
#     dataset = pd.read_csv(path, 'utf-8', header=None, names=['Dutch'], engine='python')
#     dataset = dataset.to_numpy()
#     dataset = [item for sublist in dataset for item in sublist]
#     return dataset
#
# full_dataset = import_europarl_dataset('TrainFiles-BERT-TINY/europarl-v7.nl-en.nl')
# europarl_dataset = full_dataset[20000:]
#
# print("Sample of the europarl dataset: ")
# print(europarl_dataset[0:5])
# print("amount of europarl sentences: ")
# print(len(europarl_dataset))
#
# europarl_dataset_1400k_2000k = europarl_dataset[1400000:]
# this_time = time.time()
# Tester = change_text_to_morphs(europarl_dataset_1400k_2000k,  save = True, filename = 'TrainFiles-BERT-TINY/FrogMorphedMergedEuroparl1400k_2000k.pickle', frog_merge=True)
# print(format_time(time.time() - this_time))
# print("this is time of 700k-1400k sentences merge morph process")




#NoMERGE FROG fast 50000
#0:11:52

#MERGE FROG fast 50000 europarl sentences
#0:13:51
#-> 546.9 minutes
#-> crawl = 2 times as big.. (larger sentences). -> 1093.8
#-> 1640.7 -> 27.3 uur


###Morfessor (only need to process a bit for bpe you know)
    #ucto part: 200000 sentences -> 0:12:33
    #           1000000 sentences -> 1:00:16

    ###Morfessor training:
        #1M: 20472 seconds (but smaller file is much faster I think) -> 5.69 uur
        #1M: processing file: started 10:55 of 11:00 -> 2 uur
        #500k: 4240 seconds -> 1.18 uur
        #750k: 8519 seconds -> 2.37 uur
        #50k:  623 seconds  ->

        # Epochs: 17
        # Final cost: 2467556.6891128025
        # Training time: 10818.313s
        # Saving model to 'BertTinyMorfessorModel15kNumMorphs500k.bin'...


#FROG MORPH TIMES!

#5:46:25
#First 700k

#3:37:44
#this is time of 700k-1400k sentences merge morph process

#574383
#2:48:10
#this is time of 1400k-1974383k sentences merge morph process

#####Europarl total time = 2:48:10 + 3:37:44 + 5:46:25 = 12:12:19




#
# with open("TrainFiles-BERT-TINY/nl_part_1.txt") as f:
#     lines = f.readlines()
#
# oscar_dataset_newlines = lines[:2000000]
##print(oscar_dataset_newlines[:10])
#oscar_dataset = []
#for sentence in oscar_dataset_newlines:
#    oscar_dataset.append(sentence.rstrip('\n'))

#oscar_dataset = oscar_dataset_newlines

##print("Sample of the oscar dataset: ")
##print(oscar_dataset[0:5])
##print("amount of oscar sentences: ")
##print(len(oscar_dataset))
##print("rstripped sentences: ")
##print(oscar_dataset[0].rstrip('\n'))
##print(oscar_dataset[1].rstrip('\n'))




# oscar_dataset_1800k_2000k = oscar_dataset[1800000:]
# this_time = time.time()
# Tester = change_text_to_morphs(oscar_dataset_1800k_2000k,  save = True, filename = 'TrainFiles-BERT-TINY/FrogMorphedMergedOscar1800k_2000k.pickle', frog_merge=True)
# print(format_time(time.time() - this_time))
# print("this is time of 1800k-2000k sentences merge morph process FOR OSCAR")


################################################################## VERGEET NIET rstrip(\n) nodig!!!

#12:10:30
#this is time of 300k sentences merge morph process FOR OSCAR

#12:23:32
#this is time of 300k-600k sentences merge morph process FOR OSCAR

#5:33:27
#this is time of 600k-800k sentences merge morph process FOR OSCAR

#3:24:35
#this is time of 800k-900k sentences merge morph process FOR OSCAR

#10:04:37
#this is time of 900k-1200k sentences merge morph process FOR OSCAR

#8:40:56
#this is time of 1200k-1400k sentences merge morph process FOR OSCAR

#3:49:24
#this is time of 1400k-1500k sentences merge morph process FOR OSCAR

#10:17:38
#this is time of 1500k-1800k sentences merge morph process FOR OSCAR

#8:46:27
#this is time of 1800k-2000k sentences merge morph process FOR OSCAR

#75:11
#75.18 uur

#europarl 12:12 (12.20 uur)


#-> 87:23 (87.38 uur) -> 3.64 dagen

###128 seq length

###europarl

#1000k :

#EUROPARL ENCODING TIME:
#0:09:19

#1000k-2000k :

#EUROPARL ENCODING TIME:
#0:09:06


###oscar:

#1000-1500:
#OSCAR ENCODING TIME:
#0:07:18

#1500-2000
#OSCAR ENCODING TIME:
#0:08:51



##256 seq length down


#1500k sentences
#Total fit time: 
#0:03:59
#finished fit


##Oscar

#300
#TOTAL DATASETS IMPORTING TIME:
#0:00:01

#OSCAR ENCODING TIME:
#0:05:46

#300-600
#TOTAL DATASETS IMPORTING TIME:
#0:00:02

#OSCAR ENCODING TIME:
#0:06:17

#600-900
#TOTAL DATASETS IMPORTING TIME:
#0:00:01
#OSCAR ENCODING TIME:
#0:05:47

#900-1200

#TOTAL DATASETS IMPORTING TIME:
#0:00:02
#OSCAR ENCODING TIME:
#0:06:09

#1200-1500

#TOTAL DATASETS IMPORTING TIME:
#0:00:02
#OSCAR ENCODING TIME:
#0:05:57

#1500-1800

#1800-2000
#TOTAL DATASETS IMPORTING TIME:
#0:00:02
#OSCAR ENCODING TIME:
#0:04:56

#Europarl 700k
#TOTAL DATASETS IMPORTING TIME:
#0:00:02
#EUROPARL ENCODING TIME:
#0:06:46
#EUROPARL ENCODING TIME:
#0:06:46
#EUROPARL CALCULATE SENTENCE ID LENGTH TIME:
#0:00:03

#Europarl 700k-1400k

#TOTAL DATASETS IMPORTING TIME:
#0:00:02
#EUROPARL ENCODING TIME:
#0:06:38
#EUROPARL ENCODING TIME:
#0:06:38
#EUROPARL CALCULATE SENTENCE ID LENGTH TIME:
#0:00:02

#Europarl 1400k-2000k

#TOTAL DATASETS IMPORTING TIME:
#0:00:01
#EUROPARL ENCODING TIME:
#0:05:13
#EUROPARL ENCODING TIME:
#0:05:13
#EUROPARL CALCULATE SENTENCE ID LENGTH TIME:
#0:00:02




# oldold

#TOTAL DATASETS IMPORTING TIME:
#0:00:04
#EUROPARL ENCODING TIME:
#0:11:10
#EUROPARL CALCULATE SENTENCE ID LENGTH TIME:
#0:00:03

#Europarl 1000k-2000k

#TOTAL DATASETS IMPORTING TIME:
#0:00:04
#EUROPARL ENCODING TIME: #
#0:09:32
#EUROPARL CALCULATE SENTENCE ID LENGTH TIME:
#0:00:04

#Oscar 300k

#TOTAL DATASETS IMPORTING TIME:
#0:00:01
#OSCAR ENCODING TIME:
#0:12:19

#Oscar 300k_600k

#TOTAL DATASETS IMPORTING TIME:
#0:00:01
#OSCAR ENCODING TIME:
#0:12:56


############################
#SOLELY LEMMAPOS#

with open('TrainFiles-BERT-TINY/FrogLemmaPosEuroparl700k.pickle', 'rb') as inputfile:
    FrogLemmaPosEuroparl700k = pickle.load(inputfile)

with open('TrainFiles-BERT-TINY/FrogLemmaPosEuroparl700k_1400k.pickle', 'rb') as inputfile:
    FrogLemmaPosEuroparl700k_1400k = pickle.load(inputfile)

with open('TrainFiles-BERT-TINY/FrogLemmaPosEuroparl1400k_2000k.pickle', 'rb') as inputfile:
    FrogLemmaPosEuroparl1400k_2000k = pickle.load(inputfile)

#with open('TrainFiles-BERT-TINY/FrogLemmaPosOscar100k.pickle', 'rb') as inputfile:
#    FrogLemmaPosOscar100k = pickle.load(inputfile)

#with open('TrainFiles-BERT-TINY/FrogLemmaPosOscar100k_400k.pickle', 'rb') as inputfile:
#    FrogLemmaPosOscar100k_400k = pickle.load(inputfile)

#with open('TrainFiles-BERT-TINY/FrogLemmaPosOscar400k_600k.pickle', 'rb') as inputfile:
#    FrogLemmaPosOscar400k_600k = pickle.load(inputfile)

#with open('TrainFiles-BERT-TINY/FrogLemmaPosOscar600k_1000k.pickle', 'rb') as inputfile:
#    FrogLemmaPosOscar600k_1000k = pickle.load(inputfile)

#with open('TrainFiles-BERT-TINY/FrogLemmaPosOscar1000k_1300k.pickle', 'rb') as inputfile:
#    FrogLemmaPosOscar1000k_1300k = pickle.load(inputfile)

#with open('TrainFiles-BERT-TINY/FrogLemmaPosOscar1300k_1400k.pickle', 'rb') as inputfile:
#    FrogLemmaPosOscar1300k_1400k = pickle.load(inputfile)

#with open('TrainFiles-BERT-TINY/FrogLemmaPosOscar1400k_1700k.pickle', 'rb') as inputfile:
#    FrogLemmaPosOscar1400k_1700k = pickle.load(inputfile)

#with open('TrainFiles-BERT-TINY/FrogLemmaPosOscar1700k_2000k.pickle', 'rb') as inputfile:
#    FrogLemmaPosOscar1700k_2000k = pickle.load(inputfile)


FrogLemmaPosEuroparlTotal = FrogLemmaPosEuroparl700k + FrogLemmaPosEuroparl700k_1400k + FrogLemmaPosEuroparl1400k_2000k
#FrogLemmaPosEuroparlTotal = FrogLemmaPosEuroparlTotal[1000000:]
#FrogLemmaPosOscarTotal = FrogLemmaPosOscar1000k_1300k+FrogLemmaPosOscar1300k_1400k+FrogLemmaPosOscar1400k_1700k+FrogLemmaPosOscar1700k_2000k

###################


# with open('TrainFiles-BERT-TINY/FrogMorphedMergedEuroparl700k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedEuroparl700k = pickle.load(inputfile)
#
# with open('TrainFiles-BERT-TINY/FrogMorphedMergedEuroparl700k_1400k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedEuroparl700k_1400k = pickle.load(inputfile)
#
# with open('TrainFiles-BERT-TINY/FrogMorphedMergedEuroparl1400k_2000k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedEuroparl1400k_2000k = pickle.load(inputfile)


# FrogMorphedMergedEuroparlTotal = FrogMorphedMergedEuroparl700k + FrogMorphedMergedEuroparl700k_1400k + FrogMorphedMergedEuroparl1400k_2000k
# FrogMorphedMergedEuroparlTotal = FrogMorphedMergedEuroparlTotal[1000000:]

#FrogMorphedMergedEuroparlTokenizerTrain = FrogMorphedMergedEuroparlTotal[:1000000]
#FrogMorphedMergedEuroparlTokenizerTrain = FrogMorphedMergedEuroparlTotal[:1500000]
#print("First")
#print(FrogMorphedMergedEuroparlTokenizerTrain[:5])
#print("THIS SENTECE BEFORE")
#print("")
#print(FrogMorphedMergedEuroparlTokenizerTrain[4518])
#print("")

#for i in range(len(FrogMorphedMergedEuroparlTokenizerTrain)):
#    FrogMorphedMergedEuroparlTokenizerTrain[i] = FrogMorphedMergedEuroparlTokenizerTrain[i].replace(' __add_merge__ ', ' ')
#print(FrogMorphedMergedEuroparlTokenizerTrain[:5])


#
# print("THIS SENTECE ")
# print("")
# print(FrogMorphedMergedEuroparlTokenizerTrain[4518])
# print("")

# with open('TrainFiles-BERT-TINY/FrogMorphedMergedOscar300k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedOscar300k = pickle.load(inputfile)
#
# with open('TrainFiles-BERT-TINY/FrogMorphedMergedOscar300k_600k.pickle', 'rb') as inputfile:
#    FrogMorphedMergedOscar300k_600k = pickle.load(inputfile)
#
# with open('TrainFiles-BERT-TINY/FrogMorphedMergedOscar600k_800k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedOscar600k_800k = pickle.load(inputfile)
#
# with open('TrainFiles-BERT-TINY/FrogMorphedMergedOscar800k_900k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedOscar800k_900k = pickle.load(inputfile)
#
# with open('TrainFiles-BERT-TINY/FrogMorphedMergedOscar900k_1200k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedOscar900k_1200k = pickle.load(inputfile)
#
# with open('TrainFiles-BERT-TINY/FrogMorphedMergedOscar1200k_1400k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedOscar1200k_1400k = pickle.load(inputfile)
#
# with open('TrainFiles-BERT-TINY/FrogMorphedMergedOscar1400k_1500k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedOscar1400k_1500k = pickle.load(inputfile)
#
# with open('TrainFiles-BERT-TINY/FrogMorphedMergedOscar1500k_1800k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedOscar1500k_1800k = pickle.load(inputfile)
# #
# with open('TrainFiles-BERT-TINY/FrogMorphedMergedOscar1800k_2000k.pickle', 'rb') as inputfile:
#     FrogMorphedMergedOscar1800k_2000k = pickle.load(inputfile)
# # #
# FrogMorphedMergedOscarTotal = FrogMorphedMergedOscar300k + FrogMorphedMergedOscar300k_600k + FrogMorphedMergedOscar600k_800k + FrogMorphedMergedOscar800k_900k + FrogMorphedMergedOscar900k_1200k + FrogMorphedMergedOscar1200k_1400k + FrogMorphedMergedOscar1400k_1500k + FrogMorphedMergedOscar1500k_1800k + FrogMorphedMergedOscar1800k_2000k
# FrogMorphedMergedOscarTotal = FrogMorphedMergedOscarTotal[1500000:]
# print("Oscar_sample")
# print(FrogMorphedMergedOscarTotal[:3])
#
#
#
# # for i in range(len(FrogMorphedMergedOscarTotal)):
# #     FrogMorphedMergedOscarTotal[i] = FrogMorphedMergedOscarTotal[i].replace(' __add_merge__ ', ' ')
# #     if "的" in FrogMorphedMergedOscarTotal[i] or "mileugebied" in FrogMorphedMergedOscarTotal[i]:
# #         print("HEREHEREHERE")
# #         print(FrogMorphedMergedOscarTotal[i])
# #         print(len(FrogMorphedMergedOscarTotal[i].split(' ')))
# # print(FrogMorphedMergedOscarTotal[:5])
#
# i = 0
# for sentence in FrogMorphedMergedOscarTotal:
#     if len(sentence.split(" ")) > 100000:
#        print(sentence)
#        i+=1
# print("how many too long oscar")
# print(i)
#
# i = 0
# for sentence in FrogMorphedMergedEuroparlTotal:
#     if len(sentence.split(" ")) > 1000:
#        i+=1
#        print(sentence)
# print("how many too long europarl")
# print(i)

#LemmaPosFitData = FrogLemmaPosEuroparlTotal[:1000000] + FrogLemmaPosOscarTotal[:300000]



total_importing_time = format_time(time.time() - loading_files_start_time)
creating_and_fitting_time_start = time.time()
#
encoder = Encoder(30522, pct_bpe=1, ngram_max=50)  ### params random still!

### Selftrained LEMMAPOS                    Whitespace..Whitespace
#encoder.fit(LemmaPosFitData)

#
### Selftrained MORPHMERG                    Whitespace..Whitespace
#encoder.fit(FrogMorphedMergedEuroparlTokenizerTrain)
#
#creating_and_fitting_time = format_time(time.time() - creating_and_fitting_time_start)
#print("finished fit")

#encoder.save("SavedTokenizers/FrogLemmaPos1000kEuroparl200kOscarAll.json")
#print("finished save")
#

#encoder = Encoder.load('SavedTokenizers/FrogMorph1500k.json')
#encoder = Encoder.load('SavedTokenizers/FrogLemmaPos1500k.json')
encoder = Encoder.load('SavedTokenizers/FrogLemmaPos300kEuroparl100kOscarAll.json')

print("BPE Vocab! ")
print(encoder.bpe_vocab)
print("")
print("")
print("word_vocab: ")
print(encoder.word_vocab)
print("")
print("HIGHEST VOCAB iNDEX IS")
print(max(list(encoder.bpe_vocab.values())))

# text = "Goede psychologische thriller. Prima debuut! Prachtig en indrukwekkend boek. Een aanrader! Een adembenemende pageturner! Z'n beste boek! Aangrijpend, meeslepend boek dat zo uitleest. Een prachtig boek die je razendsnel uitleest!"
#
#
# tokenized_test = encoder.tokenize(text)
#
# tokens_without_eow = []
# for token in tokenized_test:
#  if not (token is encoder.EOW or token is encoder.SOW):
#     tokens_without_eow += [token]
# print(tokens_without_eow)



########## Encoding sentences ##########





### Encode europarl sentences reviews ###

transform_start_time = time.time()

###For length purposes only:
for_length_europarl_sentences_ids = []

print("Encoding europarl sentences reviews")

i = 0
europarl_length = len(FrogLemmaPosEuroparlTotal[:1000000])
europarl_sentences_ids = []
for sentence in FrogLemmaPosEuroparlTotal[:1000000]:
    sentence_ids,_ = encoder.transform(sentence, fixed_length = fixed_len)
    if i % 1000 == 0:
        print("sentence encoded: " + str(i) + " of: " + str(europarl_length))
    i += 1
    if i % 10000 == 0:
        print("")
        print("Sentence: ")
        print(sentence)
        print("")
        print("Sentence ids: ")
        print(sentence_ids)
        #print("")
        #print("Sentence ids without padding: ")
        #print(sentence_ids_without_padding)
        print("")
        print("Sentence tokens: ")
        print(encoder.inverse_transform_list([sentence_ids]))
        print("")
    europarl_sentences_ids.append(sentence_ids)

    ###For length purposes only:
    #for_length_europarl_sentences_ids.append(sentence_ids_without_padding)


transform_europarl_time = format_time(time.time() - transform_start_time)

##################
#
# with open('TrainFiles-BERT-TINY/europarl_sentences_ids_1000k_2000k_128seq.pickle', 'wb') as outputfile:
#   pickle.dump(europarl_sentences_ids, outputfile)



# calculate_sentence_ids_lengths_europarl_start = time.time()
#
# #### FOR EUROPARL ####
#
# print("Calculating europarl sentence ids lengths")
#
# sentences_ids_len = [len(x) for x in for_length_europarl_sentences_ids]
# pd.Series(sentences_ids_len).hist()
# pd.Series(sentences_ids_len).describe().to_csv("InfoDatasetTINYBERT/europarl_1000k.csv")
# print(pd.Series(sentences_ids_len).describe())
# plt.xlabel = "Sentence Length"
# plt.ylabel = "Amount of sentences"
# #plt.legend()
# plt.savefig("InfoDatasetTINYBERT/europarl_1400k_2000k.png",format='png')
#
# total_calculate_sentence_ids_lengths_europarl_time = format_time(time.time() - calculate_sentence_ids_lengths_europarl_start)
#


# ##################
### OSCAR
# #
# transform_middle_time = time.time()
#
# ###For length purposes only:
# for_length_oscar_sentences_ids = []
#
# print("Encoding oscar sentences reviews")
#
# i = 0
# oscar_length = len(FrogLemmaPosOscarTotal)
# oscar_sentences_ids = []
# for sentence in FrogLemmaPosOscarTotal:
#     sentence_ids,sentence_ids_without_padding = encoder.transform(sentence, fixed_length=fixed_len)
#     if i % 1000 == 0:
#         print("sentence encoded: " + str(i) + " of: " + str(oscar_length))
#     i += 1
#     if i % 10000 == 0:
#         print("")
#         print("Sentence: ")
#         print(sentence)
#         print("")
#         print("Sentence ids: ")
#         print(sentence_ids)
#         print("")
#         print("Sentence ids without padding: ")
#         print(sentence_ids_without_padding)
#         print("")
#         print("Sentence tokens: ")
#         print(encoder.inverse_transform_list([sentence_ids]))
#         print("")
#
#     oscar_sentences_ids.append(sentence_ids)
#
#     ###For length purposes only:
#     for_length_oscar_sentences_ids.append(sentence_ids_without_padding)
#
#
# transform_oscar_time = format_time(time.time() - transform_middle_time)
###transform_total_time = format_time(time.time() - transform_start_time)

#### FOR OSCAR ####
    
# print("Calculating oscar sentence ids lengths")
# #
# sentences_ids_len = [len(x) for x in for_length_oscar_sentences_ids]
# pd.Series(sentences_ids_len).hist()
# pd.Series(sentences_ids_len).describe().to_csv("InfoDatasetTINYBERT/LemmaPos_oscar_1000k_2000k.csv")
# print(pd.Series(sentences_ids_len).describe())
# plt.xlabel = "Sentence Length"
# plt.ylabel = "Amount of sentences"
# plt.legend()
# plt.savefig("InfoDatasetTINYBERT/LemmaPos_oscar_1000k_2000k.png",format='png')

##total_calculate_sentence_ids_lengths_time = format_time(time.time() - calculate_sentence_ids_lengths_start)


# Save features ###

with open('TrainFiles-BERT-TINY/LLemmaPos_europarl_sentences_ids_1000k_128seq.pickle', 'wb') as outputfile:
 pickle.dump(europarl_sentences_ids, outputfile)

# with open('TrainFiles-BERT-TINY/LemmaPos_oscar_sentences_ids_1000k_128seq.pickle', 'wb') as outputfile:
#  pickle.dump(oscar_sentences_ids, outputfile)
#


print("BPE VOCAB SIZE: ")
print(encoder.bpe_vocab_size)
print(len(encoder.bpe_vocab))
print("WORD VOCAB SIZE: ")
print(encoder.word_vocab_size)
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





calculate_sentence_ids_lengths_start = time.time()

#### FOR EUROPARL ####

#print("Calculating europarl sentence ids lengths")

#sentences_ids_len = [len(x) for x in for_length_europarl_sentences_ids]
#pd.Series(sentences_ids_len).hist()
#pd.Series(sentences_ids_len).describe().to_csv("InfoDatasetTINYBERT/LemmaPos_europarl_1000k_2000k.csv")
#print(pd.Series(sentences_ids_len).describe())
#plt.xlabel = "Sentence Length"
#plt.ylabel = "Amount of sentences"
##plt.legend()
#plt.savefig("InfoDatasetTINYBERT/LemmaPos_europarl_1000k_2000k.png",format='png')

#
# #


# #### FOR OSCAR ####
#
# print("Calculating oscar sentence ids lengths")
#
# sentences_ids_len = [len(x) for x in for_length_oscar_sentences_ids]
# pd.Series(sentences_ids_len).hist()
# pd.Series(sentences_ids_len).describe().to_csv("InfoDatasetTINYBERT/oscar-morfessor-1000k1000k.csv")
# print(pd.Series(sentences_ids_len).describe())
# plt.xlabel = "Sentence Length"
# plt.ylabel = "Amount of sentences"
# #plt.legend()
# plt.savefig("InfoDatasetTINYBERT/oscar-morfessor-1000k1000k.png",format='png')
#
# ##total_calculate_sentence_ids_lengths_time = format_time(time.time() - calculate_sentence_ids_lengths_start)









########## TESTING AND PRINTING ##################


nederlandseZin = "Hey heb je je aangetekende brief wel degelijk gekregen Ik hoop van wel"
nederlandseZin2 = "In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen."
nederlandseZin3 = "Dit is een korte nederlandse zin die getokenized moet worden"
nederlandseZin4 = "Wordt deze korte nederlandse zin optimaal gesegmenteerd"
test_twitterReview_neg = 'heeeeel sleeecht nie aan te zien zeer matig'

tokenized = encoder.tokenize(nederlandseZin)
tokenized2 = encoder.tokenize(nederlandseZin2)
tokenized3 = encoder.tokenize(nederlandseZin3)
tokenized4 = encoder.tokenize(nederlandseZin4)

Ids,_ = encoder.transform(nederlandseZin)
Ids2,_ = encoder.transform(nederlandseZin2)
Ids3,_ = encoder.transform(nederlandseZin3)
Ids4,_ = encoder.transform(nederlandseZin4)


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









print("Transforming example excerpts")

europarl_excerpt = "Neemt de werkdruk bij de Commissie echt af, wanneer nationale rechters verplicht zijn tot rapportage aan de Commissie? Hoe denkt de Raad hierover en is de commissaris bereid tot grondige heroverweging van deze punten? Mijnheer de Voorzitter, voordat ik mij uitspreek over het Witboek, wil ik de rapporteur, de heer von Wogau, gelukwensen. Uit het feit dat de Fractie van de Europese Sociaal-democraten slechts Ã©Ã©n enkel amendement heeft ingediend, blijkt duidelijk dat wij het in hoge mate met zijn verslag eens zijn. Wij zijn het dus eens met het verslag, en wij zijn het tevens eens met de hoofdlijnen van het Witboek, mijnheer de commissaris. Het communautaire mededingingsrecht is sinds de inwerkingtreding van het Verdrag een fundamenteel onderdeel van het Gemeenschapsbeleid. Na ongeveer veertig jaar van kracht te zijn geweest, beginnen deze regels echter tekenen van slijtage te vertonen. Ze moeten daarom dringend worden gemoderniseerd, met name op de volgende vijf punten. In de eerste plaats, het machtigingssysteem; op de tweede plaats, de decentrale toepassing; in de derde plaats, de procedureregels; in de vierde plaats, de gerechtelijke toepassing, en in de vijfde en laatste plaats, het overmatig formalisme. Het systeem van individuele machtigingen moet dringend worden hervormd: bedrijven, wetenschappers en gespecialiseerde advocaten vragen daar eenstemmig om. Ik heb geen enkel forum van deskundigen in het mededingingsrecht bijgewoond waarop niet om een wijziging van dit systeem werd gevraagd. Een systeem dat zo weinig beslissingen oplevert als het huidige, ongeacht of die beslissingen een machtiging of een verbod inhouden, is allesbehalve functioneel. De artikelen 81.1 en 82 mogen al geruime tijd door de nationale mededingingsautoriteiten worden toegepast. Artikel 81.3 mogen zij echter niet toepassen, wat de coherente toepassing van artikel 81.1 enigszins heeft belemmerd. Het is u bekend dat er bij het Hof van Justitie momenteel twee verzoeken van twee Duitse rechtbanken voorliggen om een prejudiciÃ«le beslissing te nemen over de vraag of artikel 81.1 los van artikel 81.3 kan worden toegepast. Ook op dit punt moet er dus iets worden veranderd."
wikipedia_excerpt = "België, officieel het Koninkrijk België, is een West-Europees land dat aan de Noordzee ligt en aan Nederland, Duitsland, Luxemburg en Frankrijk grenst. Het land is 30.689 km² groot[7] en heeft een bevolking van meer dan 11,5 miljoen inwoners (ruim 6,5 miljoen in het Vlaams Gewest, 3,6 miljoen in het Waals Gewest en 1,2 miljoen in het Brussels Hoofdstedelijk Gewest).[8] De belangrijkste stad is Brussel, hoofdstad van België en tevens bestuurlijk centrum van de Europese Unie en de NAVO. Het land heeft drie officiële talen: ongeveer zestig procent van de bevolking spreekt Nederlands, vooral in Vlaanderen, veertig procent spreekt Frans, vooral in Wallonië en Brussel, en minder dan een procent spreekt Duits, in de Oostkantons. De culturele en linguïstische diversiteit van het land heeft door een opeenvolging van staatshervormingen geleid tot een complex politiek systeem, waarbij in principe de grondgebonden bevoegdheden – zoals economie, werkgelegenheid en infrastructuur – liggen bij de Gewesten (het Vlaamse, het Waalse en het Brusselse), en de persoonsgebonden materies – zoals onderwijs, cultuur en welzijn – bij de Gemeenschappen (de Vlaamse, de Franse en de Duitstalige), met een overkoepelende federale overheid voor het hele grondgebied, bevoegd voor onder meer defensie, justitie en de sociale zekerheid. België ontstond na de Belgische Revolutie in 1830 toen het zich afscheidde van het Verenigd Koninkrijk der Nederlanden, waar het sinds 1815 toe behoorde. Na de onafhankelijkheid werd de jonge natie – vooral door de ontwikkeling van een zware industrie in Wallonië – een van de voortrekkers in de Industriële Revolutie. De ontwikkeling van Vlaanderen bleef achter tot het economisch zwaartepunt naar het noorden begon te verschuiven vanaf de jaren 1960. Dat is ook de periode van de vastlegging van de taalgrens, de eerste stappen in de federalisering van het land, en van de onafhankelijkheid van de Belgische kolonie Congo en de mandaatgebieden Ruanda en Burundi. België groeide uit tot 's werelds 26ste economie, werd een van de welvarendste, meest ontwikkelde en meest geglobaliseerde landen ter wereld, en bouwde met zijn vrijemarkteconomie en een beperkte overheidsinmenging aan een uitgebreide verzorgingsstaat."
#oscar_excerpt = ' '.join(oscar_dataset[50000:50020])
#oscar_excerpt = oscar_excerpt.replace('\n','')
oscar_excerpt = "Een zin"

tokenized_europarl = encoder.tokenize(europarl_excerpt)
tokenized_wikipedia = encoder.tokenize(wikipedia_excerpt)
tokenized_oscar = encoder.tokenize(oscar_excerpt)

Ids_europarl, _ = encoder.transform(europarl_excerpt)
Ids_wikipedia, _ = encoder.transform(wikipedia_excerpt)
Ids_oscar, _ = encoder.transform(oscar_excerpt)

tokens_without_eow_europarl = []
for token in tokenized_europarl:
    if not (token is encoder.EOW or token is encoder.SOW):
        tokens_without_eow_europarl += [token]

tokens_without_eow_wikipedia = []
for token in tokenized_wikipedia:
    if not (token is encoder.EOW or token is encoder.SOW):
        tokens_without_eow_wikipedia += [token]

tokens_without_eow_oscar = []
for token in tokenized_oscar:
    if not (token is encoder.EOW or token is encoder.SOW):
        tokens_without_eow_oscar += [token]

print("")
print("")
print("europarl_excerpt")
print("")
print(europarl_excerpt)
print("")
print("TOKENS")
print("")
print(tokens_without_eow_europarl)
print("")
print("ID's")
print("")
print(Ids_europarl)
print("")
print("")

print("")
print("")
print("oscar_excerpt")
print("")
print(oscar_excerpt)
print("")
print("TOKENS")
print("")
print(tokens_without_eow_oscar)
print("")
print("ID's")
print("")
print(Ids_oscar)
print("")
print("")

print("")
print("")
print("wikipedia_excerpt")
print("")
print(wikipedia_excerpt)
print("")
print("TOKENS")
print("")
print(tokens_without_eow_wikipedia)
print("")
print("ID's")
print("")
print(Ids_wikipedia)
print("")
print("")



### PRINT TIME TAKEN ###

print("")
print("TOTAL DATASETS IMPORTING TIME: ")
print(total_importing_time)
# print("")
# print("TOTAL CREATE AND FIT TOKENIZER TIME: ")
# print(creating_and_fitting_time)
print("")
print("EUROPARL ENCODING TIME: ")
print(transform_europarl_time)
# print("")
# print("OSCAR ENCODING TIME: ")
# print(transform_oscar_time)
# print("")
# print("TOTAL ENCODING TIME: ")
# print(transform_total_time)
# print("")
# print("TOTAL CALCULATING SENTENCE LENGTHS TIME: ")
# print(total_calculate_sentence_ids_lengths_time)






# print("")
# print("EUROPARL ENCODING TIME: ")
# print(transform_europarl_time)
# print("")
#
# print("")
# print("EUROPARL CALCULATE SENTENCE ID LENGTH TIME: ")
# print(total_calculate_sentence_ids_lengths_europarl_time)
# print("")
































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



