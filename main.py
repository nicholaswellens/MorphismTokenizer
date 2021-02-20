import math

from frog import Frog, FrogOptions
from polyglot.downloader import downloader
from polyglot.text import Text, Word
import morfessor
import pickle
import re


Processed_Sentence = " lezen optimaal liep europese unie gekregen spellen rugzak super allesinds boomhut ontwikkelende gemeenschappen vermeenigvuldigde getallen Vereenvoudigd. ....... is werken lopen een kleine test gewoon om te zien of het wel werkt."
#Processed_Sentence = "Ik spring wat rond in het rond"
#frog = Frog(FrogOptions(tok=True, lemma=True, morph = True, daringmorph=False, mwu=False, chunking=True, ner=True, parser=False))
frog = Frog(FrogOptions(tok=True, lemma=True, morph=True, daringmorph=False, mwu=False, chunking=False, ner=False,parser=False))
output = frog.process(Processed_Sentence)


print("")
print("RAW OUTPUT")
print(output)
print("length")
print(len(output))
print(output[0])
print(output[1])
print(output[2])
print(output[3])

print("")
print("index:    " + str(output[0].get("index")))
print("text:     " + str(output[0].get("text")))
print("lemma:    " + str(output[0].get("lemma")))
print("pos:      " + str(output[0].get("pos")))
print("morph:    " + str(output[0].get("morph")))
print(output[0].get("morph"))
print(output[0].get("morph"))
print("chunking: " + str(output[1].get("chunker")))
print("ner:      " + str(output[0].get("ner")))
print("")
print("")

for i in range(0,len(output)):
    print(str(output[i].get("morph")))



ToDecipher = output[0].get("morph")
print(ToDecipher)

strs = ToDecipher.replace('[','').split(']')
print(strs)
print(list(filter(None, strs)))
print(type(strs))
#lists = (map(int, s.replace))


pos = output[0].get("pos")
beforeParentheses = re.findall("[A-Z]+\(",pos)
beforeParenthesesString = "".join(beforeParentheses)
purePOS = re.findall("[A-Z]+",beforeParenthesesString)

print("beforeBrackets = ")
print(beforeParentheses)
print("")
print("purePOS = ")
print(purePOS)



#print("")
#print("split")
#print("split")
#print("split")
#output = frog.process("Dit is nog een test.")
#print("PARSED OUTPUT=",output)


def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return (str1.join(s))

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



# with open('FullTrainData.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     FullTrainCorpusList = pickle.load(filehandle)
#
# full_train_text = ' '.join(FullTrainCorpusList)

with open('PosTestData.data', 'rb') as filehandle:
    # read the data as binary data stream
    PosTestCorpusList = pickle.load(filehandle)

ShorterCorpusList = PosTestCorpusList[:100]
short_text = ' '.join(ShorterCorpusList)

pos_test_text = ' '.join(PosTestCorpusList)


print(downloader.supported_languages_table("morph2"))


words = ["preprocessing", "processor", "invaluable", "thankful", "crossed"]
for w in words:
  w = Word(w, language="en")
  print("{:<20}{}".format(w, w.morphemes))



# train_data = list(pos_test_text)
#
# io = morfessor.MorfessorIO()
#
# #train_data = list(io.read_corpus_file('training_data'))
#
# model = morfessor.BaselineModel()
#
# #model.load_data(train_data, count_modifier=lambda x: 1)
# #def log_func(x):
# #    return int(round(math.log(x + 1, 2)))
# #model_logtokens.load_data(train_data, count_modifier=log_func)
# model.load_data(train_data)
#
# model.train_batch()
# #for model in models:
# #    model.train_batch()
#
# goldstd_data = io.read_annotations_file('gold_std')
# ev = morfessor.MorfessorEvaluation(goldstd_data)
# results = [ev.evaluate_model(model)]
# print("ok")
# print(results)
#
# #wsr = morfessor.WilcoxonSignedRank()
# #r = wsr.significance_test(results)
# #WilcoxonSignedRank.print_table(r)
