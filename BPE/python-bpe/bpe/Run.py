from encoder import Encoder
import pickle

#with open('morph_counts.pickle', 'rb') as inputfile:
#    morph_counts = pickle.load(inputfile)

#print("did")
#print(morph_counts)


# Generated with http://pythonpsum.com
test_corpus = '''
    Object raspberrypi functools dict kwargs. Gevent raspberrypi functools. Dunder raspberrypi decorator dict didn't lambda zip import pyramid, she lambda iterate?
    Kwargs raspberrypi diversity unit object gevent. Import fall integration decorator unit django yield functools twisted. Dunder integration decorator he she future. Python raspberrypi community pypy. Kwargs integration beautiful test reduce gil python closure. Gevent he integration generator fall test kwargs raise didn't visor he itertools...
    Reduce integration coroutine bdfl he python. Cython didn't integration while beautiful list python didn't nit!
    Object fall diversity 2to3 dunder script. Python fall for: integration exception dict kwargs dunder pycon. Import raspberrypi beautiful test import six web. Future integration mercurial self script web. Return raspberrypi community test she stable.
    Django raspberrypi mercurial unit import yield raspberrypi visual rocksdahouse. Dunder raspberrypi mercurial list reduce class test scipy helmet zip?
'''
opened =open("Old/big.txt", "r")
bigTxt =opened.read()

dutch_opened = open("EenPosReview.txt", "r")
dutchTxt = dutch_opened.read()


with open('Old/FullTrainData.data', 'rb') as filehandle:
    # read the data as binary data stream
    FullTrainCorpusList = pickle.load(filehandle)

full_train_text = ' '.join(FullTrainCorpusList)


with open('Old/PosTestData.data', 'rb') as filehandle:
    # read the data as binary data stream
    PosTestCorpusList = pickle.load(filehandle)

ShorterCorpusList = PosTestCorpusList[:100]
short_text = ' '.join(ShorterCorpusList)

pos_test_text = ' '.join(PosTestCorpusList)



encoder = Encoder(3000, pct_bpe=0.7, pct_morph=0.3, ngram_max=30)  # params chosen for demonstration purposes
encoder.fit(short_text.split('\n'))
print("finished fit")


print("BPE Vocab! ")
print(encoder.bpe_vocab)
print("")
print("")
print("Morphism Vocab! ")
print(encoder.morph_vocab)
print("")
print("")
print("word_vocab: ")
print(encoder.word_vocab)



randomEnglishSentence = "hey this is just a tokenized sentence do we need lowerlevel characters or different, who knows let's see if splelling mistakes ge interpreted"
nederlandseZin = "Hey heb je je aangetekende brief wel degelijk gekregen Ik hoop van wel"
test_twitterReview_neg = 'heeeeel sleeecht nie aan te zien zeer matig'

#print("")
#print(encoder.tokenize("dit is 雙喜 een zin die ik ga proberen tokenizen"))
#print("")

tokenized = encoder.tokenize(nederlandseZin)
#twitterized = encoder.tokenize(test_twitterReview_neg)
Ids = encoder.transform(nederlandseZin)
#twitterIds = encoder.transform(test_twitterReview_neg)


tokens_without_eow = []
for token in tokenized:
    if not (token is encoder.EOW or token is encoder.SOW):
        tokens_without_eow += [token]

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


print(encoder.bpe_vocab_size)
print(encoder.morph_vocab_size)
print(encoder.word_vocab_size)

print(len(encoder.bpe_vocab))
print(len(encoder.morph_vocab))
print(len(encoder.word_vocab))

print(encoder.bpe_vocab_size + encoder.morph_vocab_size + encoder.word_vocab_size)
print(len(encoder.bpe_vocab) + len(encoder.morph_vocab) + len(encoder.word_vocab))


print(max(list(encoder.bpe_vocab.values())))

#twitter_without_eow = []
#for token in twitterized:
#    if not (token is encoder.EOW or token is encoder.SOW):
#        twitter_without_eow += [token]

#print("")
#print("")
#print("")
#print("SENTENCE2")
#print("")
#print(test_twitterReview_neg)
#print("")
#print("TOKENS")
#print("")
#print(twitter_without_eow)
#print("")
#print("ID's")
#print("")
#print(twitterIds)


#IdsIterable = encoder._transform(nederlandseZin)
#print("TOKENS INVERSE TRANSFORMED")
#print(next(encoder.inverse_transform(IdsIterable)))
#print(encoder.transform(randomSentence))








#print("word_vocab")
#print(encoder.word_vocab)
#print("")
#print("bpe_vocab")
#print(encoder.bpe_vocab)

example = "Vizzini: He didn't fall? INCONCEIVABLE!"
#print(encoder.tokenize(example))
# ['__sow', 'vi', 'z', 'zi', 'ni', '__eow', '__sow', ':', '__eow', 'he', 'didn', "'", 't', 'fall', '__sow', '?', '__eow', '__sow', 'in', 'co', 'n', 'ce', 'iv', 'ab', 'le', '__eow', '__sow', '!', '__eow']
#print(next(encoder.transform([example])))
# [24, 108, 82, 83, 71, 25, 24, 154, 25, 14, 10, 11, 12, 13, 24, 85, 25, 24, 140, 59, 39, 157, 87, 165, 114, 25, 24, 148, 25]
#print(next(encoder.inverse_transform(encoder.transform([example]))))
# vizzini : he didn ' t fall ? inconceivable !