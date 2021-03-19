########## BPE TOKENIZER ###############


###Opmerkingen:
#  Misschien de punctuation eruit doen voor dat je morph dingen doet anders komen er (overbodig) "." "?" "..." in uw morph vocab
#  Misschien overlapping morphs en bpe's 1 van verwijderen (bv. de en de wsssss). Of enkel BPE doen als geen morph woord is natuurlijk, zo gaan doen zoals bij word_vocab
#  Geslopen -> sloopt (wordt dan mss niet gebruikt, maar niet belangrijk)
#  Maybe alot of high frequency words will dominate the list (normal)? --> enough morph percentage should fix this I guess.. FIXED: only added morphemes that have multiple in woord
   #As a result now there are gespeeld -> speel. toestanden omdat er meer dan 1 morpheme is :)

# Wilt mss niet teee veel morpheme vocab, want dan gaan er grotere woorden domineren op de kleinen. ('speel' krijgt bijvoorbeeld voorrang op 'd' omdat die langer is! maar maakt misschien niet uit he.. d moet enkel in posities zijn waar d aanwezig is en er zijn geen "gespeeld" in morph vocab..)
# Moet ik de woorden die hetzelfde in bpe als morph zijn, zelfde ID geven? NEE WANT JE GAAT TOCH ALTIJD MAAR 1 vd 2 (morph - bpe) gebruiken! Priority

#MSS slim: volledig splitsen in morphemen! (en gewoon een beetje bpe voor).. Nee, want vocab huge

#Mogelijks morphism counter sneller laten gaan door: 1 keer per woord maar te processen (nu doet alle words list), door dictionary. elk woord -> maal aantal occurences
    #(of nog meer door enkel de meest geziene woorden te ontleden, maar niet zo goed mss )


#Misschien de grootste morpheem per woord verwijderen tijdens het zoeken? zodat we enkel de essentiele kleine hebben en niet al die woorden..

#Moet denk ik echt weinig morph vocab size hebben, alle grote woorden onderaan, alle kleine delen bovenaan (meer frequent)
#MAAR KUNT MSS max aantal ngram: dus maximaal 3-4 voor een morph misschien?
#####MISSCHIEN BETER WEL grotere morphs doen, want in morph splitsing wilt ge bv denken -> 'denk-en', niet d-en-k-en. Eigenlijk wel goed zo! met veel morph woorden (zijn morphisms dus hebben die informatie van lemma in verschillende woorden). kennis van de woorden + unk door bpe opgelosd


#Twee opties denk ik: ofwel vooral bpe en dan kleine morphs gebruiken (ing, en, de) en dan proberen betere manier te fixen bij het subword-tokenizen
#    ofwel meer woorden in morph vocab doen (een beetje woord-based dan, maar gebruikt niet enkel lemma maar ook de extra kennis -d -ing -de (goeie tokens).. En geen unk problemen door de sub-word aspecten) maar bpe gebruiken voor de unk's te voorkomen en dergelijke

#heel misschien enkel morph tokenizing doen op het einde van een woord?? Want meeste morphs zijn daar, niet? Dan kan bpe de andere kant beter doen mss..
# -->> NIET WAAR [ont][wikkel][end][e], [ver][eenvoud][ig][d], etc..

#Mischien per 1000 Morphen process, zodat sneller is intotaal!
#Minder woorden bekijken voor de morphs te vinden? is optie.

#heel kleine opmerking: nu wordt morph analyse gedaan op alle woorden dat niet in word-vocab gefixed zijn.. (als word functionaliteit ooit gebruikt) mss beter om analyse op alle woorden te doen, maar dat is niet zeker..



#Single word morphs nu niet toegelaten, die zijn ook vormen die minder voorkomen in verschillende woorden he!

#die -> d-i-e OMDAT er een paar character tokens zitten in morph vocab, hoe overkomen we dit? - Single word morphs toe te laten.. (maar dan minder subword herkenning)
#                                                                                             - ...

# coding=utf-8
""" An encoder which learns byte pair encodings for white-space separated text.  Can tokenize, encode, and decode. """
from collections import Counter
import time
import datetime

try:
    from typing import Dict, Iterable, Callable, List, Any, Iterator
except ImportError:
    pass

from nltk.tokenize import wordpunct_tokenize
from frog import Frog, FrogOptions
from functools import lru_cache
import re
from tqdm import tqdm
import pickle
import math
import toolz
import json
from cachier import cachier
import datetime

DEFAULT_EOW = '__eow'
DEFAULT_SOW = '__sow'
DEFAULT_UNK = '__unk'
DEFAULT_PAD = '__pad'


class Encoder:
    """ Encodes white-space separated text using byte-pair encoding.  See https://arxiv.org/abs/1508.07909 for details.
    """

    def __init__(self, vocab_size=8192, pct_bpe=0.8, pct_morph=0.2, word_tokenizer=None,
                 silent=True, ngram_min=2, ngram_max=2, required_tokens=None,
                 strict=False, lowercase=True,
                 EOW=DEFAULT_EOW, SOW=DEFAULT_SOW, UNK=DEFAULT_UNK, PAD=DEFAULT_PAD):
        if vocab_size < 1:
            raise ValueError('vocab size must be greater than 0.')

        self.EOW = EOW
        self.SOW = SOW
        self.eow_len = len(EOW)
        self.sow_len = len(SOW)
        self.UNK = UNK
        self.PAD = PAD
        self.required_tokens = list(set(required_tokens or []).union({self.UNK, self.PAD}))
        self.vocab_size = vocab_size
        self.pct_bpe = pct_bpe
        self.pct_morph = pct_morph
        self.word_vocab_size = max([int(vocab_size * (1 - (pct_bpe+pct_morph))), len(self.required_tokens or [])])
        #self.bpe_vocab_size = int((vocab_size - self.word_vocab_size) * pct_bpe)
        self.bpe_vocab_size = int(vocab_size * pct_bpe)
        self.morph_vocab_size = vocab_size - self.word_vocab_size - self.bpe_vocab_size
        self.word_tokenizer = word_tokenizer if word_tokenizer is not None else wordpunct_tokenize
        self.custom_tokenizer = word_tokenizer is not None
        self.word_vocab = {}  # type: Dict[str, int]
        self.morph_vocab = {}  # type: Dict[str, int]
        self.bpe_vocab = {}  # type: Dict[str, int]
        self.inverse_word_vocab = {}  # type: Dict[int, str]
        self.inverse_bpe_vocab = {}  # type: Dict[int, str]
        self._progress_bar = iter if silent else tqdm
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.strict = strict
        self.lowercase = lowercase

    def mute(self):
        """ Turn on silent mode """
        self._progress_bar = iter

    def unmute(self):
        """ Turn off silent mode """
        self._progress_bar = tqdm

    # ---
    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))


    def byte_pair_counts(self, words):
        # type: (Encoder, Iterable[str]) -> Iterable[Counter]
        """ Counts space separated token character pairs:
            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4}
        """
        for token, count in self._progress_bar(self.count_tokens(words).items()):
            bp_counts = Counter()  # type: Counter
            for ngram in token.split(' '):
                bp_counts[ngram] += count
            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)]) + 1):
                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(ngram_size, token.split(' '))]

                for ngram in ngrams:
                    bp_counts[''.join(ngram)] += count

            yield bp_counts



    def morph_counts_old_version(self, words):
        #Word List to list of all morphisms
        print("len words: ")
        print(len(words))
        print("len unique words: ")
        print(len(set(words)))
        frog = Frog(FrogOptions(tok=True, lemma=True, morph=True, daringmorph=False, mwu=False, chunking=False, ner=False,parser=False))
        morphisms = []
        print_counter = 1
        t0 = time.time()
        for word in words:
            output = frog.process(word)
            morphisms_word = output[0].get("morph")
            morphisms_word_list = morphisms_word.replace('[', '').split(']')
            #Momenteel GEEN GEHELE WOORDEN IN COUNT
            if len(morphisms_word_list) > 2:
                morphisms += morphisms_word_list
            total_length = len(words)
            print(str(print_counter) + " of " + str(total_length))
            print_counter += 1
        print("Frog Processing Time:")
        print(self.format_time(time.time() - t0))



        morphisms = list(filter(None, morphisms))
        morph_counts = Counter(morphisms)
        return morph_counts

    def morph_counts_new_version(self, words):
        #Word List to list of all morphisms
        frog = Frog(FrogOptions(tok=True, lemma=True, morph=True, daringmorph=False, mwu=False, chunking=False, ner=False,parser=False))
        words_string = ' '.join(words)
        morphisms = []
        print_counter = 1
        t0 = time.time()
        print("Starting Frog Processing..")
        output = frog.process(words_string)
        print("Process time:")
        process_time = self.format_time(time.time() - t0)
        print(process_time)
        t1 = time.time()
        for i in range(0,len(words)-1):
            morphisms_word = output[i].get("morph")
            morphisms_word_list = morphisms_word.replace('[', '').split(']')
            #Momenteel GEEN GEHELE WOORDEN IN COUNT
            if len(morphisms_word_list) > 2:
                morphisms += morphisms_word_list
            total_length = len(words)
            print(str(print_counter) + " of " + str(total_length))
            print_counter += 1
        print("Process Time:")
        print(process_time)
        print("Getting Morphisms Time:")
        print(self.format_time(time.time() - t1))
        print("Total Time:")
        print(self.format_time(time.time() - t0))



        morphisms = list(filter(None, morphisms))
        morph_counts = Counter(morphisms)
        return morph_counts

    def morph_counts_faster_version(self, words):
        #Word List to list of all morphisms

        frog = Frog(FrogOptions(tok=True, lemma=True, morph=True, daringmorph=False, mwu=False, chunking=False, ner=False,parser=False))
        batch_size = 400
        morphisms = []
        print_batch_number = 1
        start_time = time.time()
        total_batch_number = math.ceil(len(words)/batch_size)
        total_process_time = 0
        total_getting_morphisms_time = 0
        for i in range(0, len(words), batch_size):
            t0 = time.time()
            #print_counter = 1
            words_batch = words[i:i + batch_size]
            words_batch_string = ' '.join(words_batch)
            #print("Starting Frog Processing.. for batch = " + str(print_batch_number))
            output = frog.process(words_batch_string)
            #print("Process time:")
            process_time = time.time() - t0
            #print(self.format_time(process_time))
            #print(process_time)
            t1 = time.time()
            for j in range(0,len(words_batch)-1):
                morphisms_word = output[j].get("morph")
                morphisms_word_list = morphisms_word.replace('[', '').split(']')
                #Momenteel GEEN GEHELE WOORDEN IN COUNT
                if len(morphisms_word_list) > 2:
                    morphisms += morphisms_word_list
                total_batch_length = len(words_batch)
                #print(str(print_counter) + " of " + str(total_batch_length) + " -- of batch -- " + str(print_batch_number) + " of " + str(total_batch_number) )
                #print("batch" + " (batch_size: " + str(batch_size) + " words):    " +  str(print_batch_number) + " of " + str(total_batch_number))
                #print_counter += 1
            print_batch_number += 1
            getting_morphisms_time = time.time() - t1
            total_process_time += process_time
            total_getting_morphisms_time += getting_morphisms_time

        print("Total number of words: ")
        print(len(words))
        print("")
        print("Unique number words: ")
        print(len(set(words)))
        print("")
        print("Total Process Time:")
        print(self.format_time(total_process_time))
        print("")
        print("Total Getting Morphisms Time: ")
        print(self.format_time(total_getting_morphisms_time))
        print("")
        print("Total Time:")
        print(self.format_time(time.time() - start_time))
        print("")

        morphisms = list(filter(None, morphisms))
        morph_counts = Counter(morphisms)
        return morph_counts


    #@cachier(stale_after=datetime.timedelta(days=3))
    #@lru_cache(maxsize=None)
    def morph_counts_fastest_version(self, words):
        # Word List to list of all morphisms

        word_counts = Counter(word for word in toolz.concat(map(self.word_tokenizer, words)))

        #print("words_counts: ")
        #print(word_counts)
        print("")
        print("Unique number words: " + str(len(set(words))))
        print("Total number of words: " + str(len(words)))
        print("")

        unique_words_set = set(words)
        unique_words = list(unique_words_set)

        frog = Frog(
            FrogOptions(tok=True, lemma=True, morph=True, daringmorph=False, mwu=False, chunking=False, ner=False,
                        parser=False))
        batch_size = 400
        morphisms = []
        print_batch_number = 1
        start_time = time.time()
        total_batch_number = math.ceil(len(unique_words) / batch_size)
        total_process_time = 0
        total_getting_morphisms_time = 0

        for i in range(0, len(unique_words), batch_size):
            t0 = time.time()
            words_batch = unique_words[i:i + batch_size]
            words_batch_string = ' '.join(words_batch)
            output = frog.process(words_batch_string)
            process_time = time.time() - t0
            t1 = time.time()

            for j in range(0, len(words_batch) - 1):
                current_word = output[j].get("text")
                morphisms_word = output[j].get("morph")
                morphisms_word_list = morphisms_word.replace('[', '').split(']')
                current_word_count = word_counts[current_word]

                # Momenteel GEEN GEHELE WOORDEN IN COUNT
                if len(morphisms_word_list) > 2:
                    morphisms += morphisms_word_list * current_word_count

                total_batch_length = len(words_batch)
            print("batch" + " (" + str(batch_size) + " words):    " +  str(print_batch_number) + " of " + str(total_batch_number))

            print_batch_number += 1
            getting_morphisms_time = time.time() - t1
            total_process_time += process_time
            total_getting_morphisms_time += getting_morphisms_time

        print("")
        print("Total number of words: ")
        print(len(words))
        print("")
        print("Unique number words: ")
        print(len(set(words)))
        print("")
        print("Total Process Time:")
        print(self.format_time(total_process_time))
        print("")
        print("Total Getting Morphisms Time: ")
        print(self.format_time(total_getting_morphisms_time))
        print("")
        print("Total Time:")
        print(self.format_time(time.time() - start_time))
        print("")

        # Remove the empty strings
        morphisms = list(filter(None, morphisms))
        #Make a counter of all morphisms
        morph_counts = Counter(morphisms)

        with open('Old/morph_counts.pickle', 'wb') as outputfile:
            pickle.dump(morph_counts, outputfile)

        return morph_counts



    def count_tokens(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Count tokens into a BPE vocab """
        token_counts = Counter(self._progress_bar(words))
        return {' '.join(token): count for token, count in token_counts.items()}

    def learn_word_vocab(self, sentences):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Build vocab from self.word_vocab_size most common tokens in provided sentences """
        word_counts = Counter(word for word in toolz.concat(map(self.word_tokenizer, sentences)))
        #print("word_counts")
        #print(word_counts)

        for token in set(self.required_tokens or []):
            word_counts[token] = int(2 ** 63)
            if token is '__pad':
              word_counts[token] = int(2**63)
        sorted_word_counts = sorted(word_counts.items(), key=lambda p: -p[1])
        #print("sorted_word_counts")
        #print(sorted_word_counts)
        return {word: idx for idx, (word, count) in enumerate(sorted_word_counts[:self.word_vocab_size])}

    def learn_bpe_vocab(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Learns a vocab of byte pair encodings """
        vocab = Counter()  # type: Counter
        for token in {self.SOW, self.EOW}:
            vocab[token] = int(2 ** 63)
        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):
            for byte_pair, count in byte_pair_count.items():
                vocab[byte_pair] += count

            if (idx + 1) % 10000 == 0:
                self.trim_vocab(10 * self.bpe_vocab_size, vocab)

        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:self.bpe_vocab_size]
        return {bp: idx + self.word_vocab_size + self.morph_vocab_size for idx, (bp, count) in enumerate(sorted_bpe_counts)}

    def learn_morph_vocab(self, words):

        with open('Old/morph_counts.pickle', 'rb') as inputfile:
            morph_counts = pickle.load(inputfile)

        sorted_morph_counts = sorted(morph_counts.items(), key=lambda p: -p[1])
        return {morph: idx + self.word_vocab_size for idx, (morph, count) in enumerate(sorted_morph_counts[:self.morph_vocab_size])}


    def fit(self, text):
        # type: (Encoder, Iterable[str]) -> None
        """ Learn vocab from text. """

        start_time = time.time()

        if self.lowercase:
            _text = [l.lower().strip() for l in text]
        else:
            _text = [l.strip() for l in text]
        # First, learn word vocab
        self.word_vocab = self.learn_word_vocab(_text)

        remaining_words = [word for word in toolz.concat(map(self.word_tokenizer, _text))
                           if word not in self.word_vocab]
        self.bpe_vocab = self.learn_bpe_vocab(remaining_words)
        self.morph_vocab = self.learn_morph_vocab(remaining_words)


        self.inverse_word_vocab = {idx: token for token, idx in self.word_vocab.items()}
        self.inverse_bpe_vocab = {idx: token for token, idx in self.bpe_vocab.items()}
        self.inverse_morph_vocab = {idx: token for token, idx in self.morph_vocab.items()}

        print("Total fit time: ")
        print(self.format_time(time.time()-start_time))

    @staticmethod
    def trim_vocab(n, vocab):
        # type: (int, Dict[str, int]) -> None
        """  Deletes all pairs below 10 * vocab size to prevent memory problems """
        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
        pairs_to_trim = [pair for pair, count in pair_counts[n:]]
        for pair in pairs_to_trim:
            del vocab[pair]

    def subword_tokenize_bpe(self, word):
        # type: (Encoder, str) -> List[str]
        """ Tokenizes inside an unknown token using BPE """
        end_idx = min([len(word), self.ngram_max])
        sw_tokens = []
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            #print(len(subword))
            if subword in self.bpe_vocab:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            elif len(subword) == 1:
                sw_tokens.append(self.UNK)
                #print(sw_tokens)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            else:
                end_idx -= 1
        return sw_tokens

    def subword_tokenize(self, word):
        # type: (Encoder, str) -> List[str]
        """ Tokenizes inside an unknown token using Morphisms and BPE
            Morphisms get priority over BPE tokens"""
        end_idx = min([len(word), self.ngram_max])
        sw_tokens = [self.SOW]
        #sw_tokens = []
        start_idx = 0

        #print("SUBWORD TOKENIZING WORD")

        unfound_subword = ''
        #::: unfound_subwords = []

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self.morph_vocab:
                #Subwords not found in morph_vocab get processed by BPE tokenization
                sw_tokens += self.subword_tokenize_bpe(unfound_subword)
                #Add the matching morph token
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
                #::: unfound_subwords += [unfound_subword]
                unfound_subword = ''

            elif len(subword) == 1:
                unfound_subword += subword
                #print("UNFOUND")
                #print(unfound_subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            else:
                end_idx -= 1

        #:::unfound_subwords += [unfound_subword]

        #Last unfound subword get's processed by BPE tokenization
        sw_tokens += self.subword_tokenize_bpe(unfound_subword)

        sw_tokens.append(self.EOW)
        return sw_tokens

    def tokenize(self, sentence):
        # type: (Encoder, str) -> List[str]
        """ Split a sentence into word and subword tokens """
        if self.lowercase:
            word_tokens = self.word_tokenizer(sentence.lower().strip())
        else:
            word_tokens = self.word_tokenizer(sentence.strip())
        tokens = []
        for word_token in word_tokens:
            if word_token in self.word_vocab:
                tokens.append(word_token)
            else:
                tokens.extend(self.subword_tokenize(word_token))

        return tokens

    def _transform(self, sentences, reverse=False, fixed_length=None):
        # type: (Encoder, str, bool, int) -> Iterable[List[int]]
        """ Turns space separated tokens into vocab idxs Helper function"""

        # srt -> Iterable[str]
        sentences = [sentences]

        direction = -1 if reverse else 1
        for sentence in self._progress_bar(sentences):
            in_subword = False
            encoded = []
            if self.lowercase:
                tokens = list(self.tokenize(sentence.lower().strip()))
            else:
                tokens = list(self.tokenize(sentence.strip()))
            for token in tokens:
                if in_subword:
                    if token in self.morph_vocab:
                        encoded.append(self.morph_vocab[token])
                    elif token in self.bpe_vocab:
                        if token == self.EOW:
                            in_subword = False
                        encoded.append(self.bpe_vocab[token])
                    else:
                        encoded.append(self.word_vocab[self.UNK])
                else:
                    if token == self.SOW:
                        in_subword = True
                        encoded.append(self.bpe_vocab[token])
                    else:
                        if token in self.word_vocab:
                            encoded.append(self.word_vocab[token])
                        else:
                            encoded.append(self.word_vocab[self.UNK])

            ###PADDING HIER NOG AAN RECHTERKANT
            if fixed_length is not None:
                encoded = encoded[:fixed_length]
                while len(encoded) < fixed_length:
                    encoded.append(self.word_vocab[self.PAD])

            yield encoded[::direction]

    ###

    def transform(self, sentences, reverse=False, fixed_length=None, sow_and_eow=False, right_padding=False):

        if sow_and_eow is True:

            if right_padding is True:
                # With right padding
                ID_encoding = next(self._transform(sentences, reverse, fixed_length))
            else:
                # Without padding
                ID_encoding = next(self._transform(sentences, reverse))

                # Adding left padding
                padding_list = []
                if fixed_length is not None:
                    ID_encoding = ID_encoding[:fixed_length]
                    while (len(ID_encoding) + len(padding_list)) < fixed_length:
                        padding_list.append(self.word_vocab[self.PAD])
                    ID_encoding = padding_list + ID_encoding

        else:

            if right_padding is True:
                # with right padding
                ID_encoding = next(self._transform(sentences, reverse, fixed_length))

                # Removing eow and sow id's
                eow_ID = self.bpe_vocab['__eow']
                sow_ID = self.bpe_vocab['__sow']
                ID_encoding = [x for x in ID_encoding if not (x == eow_ID or x == sow_ID)]

            else:
                # Without padding
                ID_encoding = next(self._transform(sentences, reverse))

                # Removing eow and sow id's
                eow_ID = self.bpe_vocab['__eow']
                sow_ID = self.bpe_vocab['__sow']
                ID_encoding = [x for x in ID_encoding if not (x == eow_ID or x == sow_ID)]

                # Adding left padding
                padding_list = []
                if fixed_length is not None:
                    ID_encoding = ID_encoding[:fixed_length]
                    while (len(ID_encoding) + len(padding_list)) < fixed_length:
                        padding_list.append(self.word_vocab[self.PAD])
                    ID_encoding = padding_list + ID_encoding

        return ID_encoding

    #Still needs some work
    def inverse_transform(self, rows):
        # type: (Encoder, Iterable[List[int]]) -> Iterator[str]
        """ Turns token indexes back into space-joined text. """
        for row in rows:
            words = []

            rebuilding_word = False
            current_word = ''
            for idx in row:
                if self.inverse_bpe_vocab.get(idx) == self.SOW:
                    if rebuilding_word and self.strict:
                        raise ValueError('Encountered second SOW token before EOW.')
                    rebuilding_word = True

                elif self.inverse_bpe_vocab.get(idx) == self.EOW:
                    if not rebuilding_word and self.strict:
                        raise ValueError('Encountered EOW without matching SOW.')
                    rebuilding_word = False
                    words.append(current_word)
                    current_word = ''

                elif rebuilding_word and (idx in self.inverse_bpe_vocab):
                    current_word += self.inverse_bpe_vocab[idx]

                elif rebuilding_word and (idx in self.inverse_word_vocab):
                    current_word += self.inverse_word_vocab[idx]

                elif idx in self.inverse_word_vocab:
                    words.append(self.inverse_word_vocab[idx])

                elif idx in self.inverse_bpe_vocab:
                    if self.strict:
                        raise ValueError("Found BPE index {} when not rebuilding word!".format(idx))
                    else:
                        words.append(self.inverse_bpe_vocab[idx])

                elif idx in self.inverse_morph_vocab:
                    if self.strict:
                        raise ValueError("Found Morph index {} when not rebuilding word!".format(idx))
                    else:
                        words.append(self.inverse_morph_vocab[idx])

                else:
                    raise ValueError("Got index {} that was not in word, BPE or Morph vocabs!".format(idx))

            yield ' '.join(w for w in words if w != '')

    def vocabs_to_dict(self, dont_warn=False):
        # type: (Encoder, bool) -> Dict[str, Dict[str, int]]
        """ Turns vocab into dict that is json-serializeable """
        if self.custom_tokenizer and not dont_warn:
            print("WARNING! You've specified a non-default tokenizer.  You'll need to reassign it when you load the "
                  "model!")
        return {
            'byte_pairs': self.bpe_vocab,
            'words': self.word_vocab,
            'kwargs': {
                'vocab_size': self.vocab_size,
                'pct_bpe': self.pct_bpe,
                'silent': self._progress_bar is iter,
                'ngram_min': self.ngram_min,
                'ngram_max': self.ngram_max,
                'required_tokens': self.required_tokens,
                'strict': self.strict,
                'EOW': self.EOW,
                'SOW': self.SOW,
                'UNK': self.UNK,
                'PAD': self.PAD,
            }
        }

    def save(self, outpath, dont_warn=False):
        # type: (Encoder, str, bool) -> None
        """ Serializes and saves encoder to provided path """
        with open(outpath, 'w') as outfile:
            json.dump(self.vocabs_to_dict(dont_warn), outfile)

    @classmethod
    def from_dict(cls, vocabs):
        # type: (Any, Dict[str, Dict[str, int]]) -> Encoder
        """ Load encoder from dict produced with vocabs_to_dict """
        encoder = Encoder(**vocabs['kwargs'])
        encoder.word_vocab = vocabs['words']
        encoder.bpe_vocab = vocabs['byte_pairs']

        encoder.inverse_bpe_vocab = {v: k for k, v in encoder.bpe_vocab.items()}
        encoder.inverse_word_vocab = {v: k for k, v in encoder.word_vocab.items()}

        return encoder

    @classmethod
    def load(cls, in_path):
        # type: (Any, str) -> Encoder
        """ Loads an encoder from path saved with save """
        with open(in_path) as infile:
            obj = json.load(infile)
        return cls.from_dict(obj)