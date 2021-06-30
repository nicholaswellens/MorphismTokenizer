########## BPE TOKENIZER ###############


###Opmerkingen:

# Punctuation en hoofdletters (wss niet) aanpassen!



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
from nltk.tokenize import WhitespaceTokenizer
from frog import Frog, FrogOptions
import ucto
import re
from tqdm import tqdm
import math
import toolz
import json

DEFAULT_EOW = '__eow'
DEFAULT_SOW = '__sow'
DEFAULT_UNK = '__unk'
DEFAULT_PAD = '__pad'
DEFAULT_MASK = '__mask'
DEFAULT_CLS = '__cls'
DEFAULT_SEP = '__sep'
DEFAULT_MERGE = '__merge'

configurationFile = "tokconfig-nld"
tokenizer = ucto.Tokenizer(configurationFile)


def ucto_tokenize(sentence):

    tokenized_sentence = []
    tokenizer.process(sentence)
    for token in tokenizer:
      tokenized_sentence += [str(token)]
    ucto_tokenize.counter += 1
    print(int(ucto_tokenize.counter))
    return tokenized_sentence
ucto_tokenize.counter = 0



class Encoder:
    """ Encodes white-space separated text using byte-pair encoding.  See https://arxiv.org/abs/1508.07909 for details.
    """

    def __init__(self, vocab_size=30522, pct_bpe=1, word_tokenizer=None,
                 silent=True, ngram_min=2, ngram_max=2, required_tokens=None,
                 strict=False, lowercase=True,
             EOW=DEFAULT_EOW, SOW=DEFAULT_SOW, UNK=DEFAULT_UNK, PAD=DEFAULT_PAD, MASK=DEFAULT_MASK, CLS=DEFAULT_CLS, SEP=DEFAULT_SEP, MERGE = DEFAULT_MERGE):
        if vocab_size < 1:
            raise ValueError('vocab size must be greater than 0.')

        self.EOW = EOW
        self.SOW = SOW
        self.eow_len = len(EOW)
        self.sow_len = len(SOW)
        self.UNK = UNK
        self.PAD = PAD
        self.MERGE = MERGE
        self.cls_token = CLS
        self.sep_token = SEP
        self.mask_token = MASK
        self.required_tokens = list(set(required_tokens or []).union({self.UNK, self.PAD, self.MERGE, self.mask_token,self.cls_token,self.sep_token}))
        self.vocab_size = vocab_size
        self.pct_bpe = pct_bpe
        self.word_vocab_size = max([int(vocab_size * (1 - pct_bpe)), len(self.required_tokens or [])])
        self.bpe_vocab_size = vocab_size - self.word_vocab_size
        self.word_tokenizer = word_tokenizer if word_tokenizer is not None else WhitespaceTokenizer().tokenize#ucto_tokenize#WhitespaceTokenizer().tokenize#ucto_tokenize#WhitespaceTokenizer().tokenize#ucto_tokenize #ucto_tokenize # WhitespaceTokenizer().tokenize #wordpunct_tokenize
        self.word_tokenizer_fitting = WhitespaceTokenizer().tokenize #ucto_tokenize#WhitespaceTokenizer().tokenize#ucto_tokenize#ucto_tokenize#WhitespaceTokenizer().tokenize #WhitespaceTokenizer().tokenize #wordpunct_tokenize
        self.custom_tokenizer = word_tokenizer is not None
        self.word_vocab = {}  # type: Dict[str, int]
        self.bpe_vocab = {}  # type: Dict[str, int]
        self.inverse_word_vocab = {}  # type: Dict[int, str]
        self.inverse_bpe_vocab = {}  # type: Dict[int, str]
        self._progress_bar = iter if silent else tqdm
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.strict = strict
        self.lowercase = lowercase

    def __len__(self):
        return self.vocab_size

    def mute(self):
        """ Turn on silent mode """
        self._progress_bar = iter

    def unmute(self):
        """ Turn off silent mode """
        self._progress_bar = tqdm

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



    # ---
    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def count_tokens(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Count tokens into a BPE vocab """
        token_counts = Counter(self._progress_bar(words))
        return {' '.join(token): count for token, count in token_counts.items()}

    def learn_word_vocab(self, sentences):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Build vocab from self.word_vocab_size most common tokens in provided sentences """
        word_counts = Counter(word for word in toolz.concat(map(self.word_tokenizer_fitting, sentences)))
        #print("word_counts")
        #print(word_counts)

        for token in set(self.required_tokens or []):
            word_counts[token] = int(2 ** 57)
            if token is self.PAD:
              word_counts[token] = int(2**63)
            if token is self.UNK:
              word_counts[token] = int(2**62)
            if token is self.sep_token:
              word_counts[token] = int(2**61)
            if token is self.cls_token:
              word_counts[token] = int(2**60)
            if token is self.mask_token:
              word_counts[token] = int(2**59)
            if token is self.MERGE:
              word_counts[token] = int(2**58)

        sorted_word_counts = sorted(word_counts.items(), key=lambda p: -p[1])
        return {word: idx for idx, (word, count) in enumerate(sorted_word_counts[:self.word_vocab_size])}

    def learn_bpe_vocab(self, words):
        # type: (Encoder, Iterable[str]) -> Dict[str, int]
        """ Learns a vocab of byte pair encodings """
        vocab = Counter()  # type: Counter
        for token in {self.SOW, self.EOW}:
            vocab[token] = int(2 ** 63)
        print("w")
        i = 10000
        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):

            for byte_pair, count in byte_pair_count.items():
                vocab[byte_pair] += count

            if (idx + 1) % 10000 == 0:
                print(i)
                i += 10000
                self.trim_vocab(10 * self.bpe_vocab_size, vocab)
        print("ww")
        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:self.bpe_vocab_size]
        return {bp: idx + self.word_vocab_size for idx, (bp, count) in enumerate(sorted_bpe_counts)}

    def fit(self, text):
        # type: (Encoder, Iterable[str]) -> None
        """ Learn vocab from text. """

        start_time = time.time()
        print("First")
        if self.lowercase:
            _text = [l.lower().strip() for l in text]
        else:
            _text = [l.strip() for l in text]



        # First, learn word vocab
        print("Second")
        self.word_vocab = self.learn_word_vocab(_text)
        print("Third")
        remaining_words = [word for word in toolz.concat(map(self.word_tokenizer_fitting, _text))
                           if word not in self.word_vocab]
        self.bpe_vocab = self.learn_bpe_vocab(remaining_words)
        print("Fifth")
        self.inverse_word_vocab = {idx: token for token, idx in self.word_vocab.items()}
        self.inverse_bpe_vocab = {idx: token for token, idx in self.bpe_vocab.items()}

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

    def subword_tokenize(self, word):
        # type: (Encoder, str) -> List[str]
        """ Tokenizes inside an unknown token using BPE """
        end_idx = min([len(word), self.ngram_max])
        sw_tokens = [self.SOW]
        start_idx = 0

        while start_idx < len(word):



            subword = word[start_idx:end_idx]
            if subword in self.bpe_vocab:

                # Adding MERGE Tokens inbetween
                if start_idx > 0:
                    sw_tokens.append(self.MERGE)

                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            elif len(subword) == 1:

                # Adding MERGE Tokens inbetween
                if start_idx > 0:
                    sw_tokens.append(self.MERGE)

                sw_tokens.append(self.UNK)
                start_idx = end_idx
                end_idx = min([len(word), start_idx + self.ngram_max])
            else:
                end_idx -= 1

        sw_tokens.append(self.EOW)
        return sw_tokens

    def tokenize(self, sentence, frog_merge = False):
        # type: (Encoder, str) -> List[str]
        """ Split a sentence into word and subword tokens """
        if self.lowercase:
            word_tokens = self.word_tokenizer(sentence.lower().strip())
        else:
            word_tokens = self.word_tokenizer(sentence.strip())
        tokens = []
        for word_token in word_tokens:

            if frog_merge is True:
                if word_token == "__add_merge__":
                    tokens.append(self.MERGE)
                    continue


            if word_token in self.word_vocab:
                tokens.append(word_token)
            else:
                tokens.extend(self.subword_tokenize(word_token))
        return tokens

    def _transform(self, sentences, reverse=False, fixed_length=None, frog_merge = False):
        # type: (Encoder, str, bool, int) -> Iterable[List[int]]
        """ Turns space separated tokens into vocab idxs Helper function"""

        # srt -> Iterable[str]
        sentences = [sentences]

        direction = -1 if reverse else 1
        for sentence in self._progress_bar(sentences):
            in_subword = False
            encoded = []
            if self.lowercase:
                tokens = list(self.tokenize(sentence.lower().strip(), frog_merge = frog_merge))
            else:
                tokens = list(self.tokenize(sentence.strip(), frog_merge = frog_merge))

            encoded.append(self.word_vocab[self.cls_token])

            for token in tokens:
                if in_subword:
                    if token in self.bpe_vocab:
                        if token == self.EOW:
                            in_subword = False
                        encoded.append(self.bpe_vocab[token])
                    else:
                        if token == self.MERGE:
                            encoded.append(self.word_vocab[self.MERGE])
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




            # REMOVE EOW SOW
            eow_ID = self.bpe_vocab[self.EOW]
            sow_ID = self.bpe_vocab[self.SOW]
            encoded = [x for x in encoded if not (x == eow_ID or x == sow_ID)]

            ## Purely for length purposes
            encoded_without_padding = list(encoded)
            encoded_without_padding.append(self.word_vocab[self.sep_token])

            ### Right side padding (or possible truncation)
            if fixed_length is not None:
                encoded = encoded[:(fixed_length-1)]
                encoded.append(self.word_vocab[self.sep_token])
                while len(encoded) < fixed_length:
                    encoded.append(self.word_vocab[self.PAD])
            else:
                encoded.append(self.word_vocab[self.sep_token])

            yield encoded[::direction], encoded_without_padding[::direction]

    ###

    def transform(self, sentences, reverse=False, fixed_length=None,frog_merge = False):

        ID_encoding, ID_encoding_without_padding = next(self._transform(sentences, reverse, fixed_length, frog_merge = frog_merge))
        return ID_encoding, ID_encoding_without_padding

    def inverse_transform_list(self, rows):
        for row in rows:
            tokens = []
            for id in row:
                if id in self.inverse_bpe_vocab:
                    tokens.append(self.inverse_bpe_vocab[id])
                if id in self.inverse_word_vocab:
                    tokens.append(self.inverse_word_vocab[id])
        return tokens

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

                else:
                    raise ValueError("Got index {} that was not in word or BPE vocabs!".format(idx))

            yield ' '.join(w for w in words if w != '')

    # For hugging face sync
    # Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the vocabulary.
    def convert_tokens_to_ids(self, tokens):

        if type(tokens) == str:
            token = tokens
            if token in self.word_vocab:
                ids = self.word_vocab[token]
            elif token in self.bpe_vocab:
                ids = self.bpe_vocab[token]
            else:
                ids = self.word_vocab[self.UNK]
        else:

            ids = []
            for token in tokens:
                if token in self.word_vocab:
                    ids.append(self.word_vocab[token])
                elif token in self.bpe_vocab:
                    ids.append(self.bpe_vocab[token])
                else:
                    ids.append(self.word_vocab[self.UNK])
        return ids

    # self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):

        # This function is only used in DataLoader, so exact usage (as in BERT) of this function is not required I think
        # Look to mask_tokens: https://github.com/huggingface/transformers/blob/09a2f40684f77e62d0fd8485fe9d2d610390453f/src/transformers/data/data_collator.py#L88
        # In BERT model would be different, but now it's developped for case of own tokenizer.. Normally padding is not a special token, but in my case it should be.
        # Padding in Dataloader: tokenizer._pad_token and tokenizer._pad_token_id
        # Also: SEP and CLS are ADDED to the special mask in the hugging face implementation, however here the arguments token_ids already have them inside, so we need to add
        # (No need to add Mask token here because this function is used to ADD the mask tokens appropriately)
        # MERGE: possibly add, experiments

        special_tokens_mask = []
        if already_has_special_tokens:
            for id in token_ids_0:
                if id == (self.word_vocab[self.PAD] or self.bpe_vocab[self.SOW] or self.bpe_vocab[self.EOW] or
                          self.word_vocab[self.cls_token] or self.word_vocab[self.sep_token] or self.word_vocab[self.MERGE]):
                    special_tokens_mask += [1]
                else:
                    special_tokens_mask += [0]

        else:
            raise ValueError(
                "already_has_special_tokens is turned to False, and is not adequately implemented in sync with huggingface library")
            len_token_ids = len(token_ids_0)
            special_tokens_mask = len_token_ids * [0]

        return special_tokens_mask

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
                'PAD': self.PAD
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
