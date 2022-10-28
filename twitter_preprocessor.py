import string

import nltk
from nltk.corpus import stopwords
from nltk import re

MIN_YEAR = 1900
MAX_YEAR = 2100


def get_url_patern():
    return re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))'
        r'[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})')


def get_emojis_pattern():
    try:
        # UCS-4
        emojis_pattern = re.compile(
            u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    except re.error:
        # UCS-2
        emojis_pattern = re.compile(
            u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
    return emojis_pattern


def get_hashtags_pattern():
    return re.compile(r'#\w*')


def get_single_letter_words_pattern():
    return re.compile(r'(?<![\w\-])\w(?![\w\-])')


def get_blank_spaces_pattern():
    return re.compile(r'\s{2,}|\t')


def get_twitter_reserved_words_pattern():
    return re.compile(r'(RT|rt|FAV|fav|VIA|via)')


def get_mentions_pattern():
    return re.compile(r'@\w*')


def get_negations_pattern():
    negations_ = {
        "isn't": "is not",
        "can't": "can not",
        "couldn't": "could not",
        "hasn't": "has not",
        "hadn't": "had not",
        "won't": "will not",
        "wouldn't": "would not",
        "aren't": "are not",
        "haven't": "have not",
        "doesn't": "does not",
        "didn't": "did not",
        "don't": "do not",
        "shouldn't": "should not",
        "wasn't": "was not",
        "weren't": "were not",
        "mightn't": "might not",
        "mustn't": "must not"}
    return re.compile(r'\b(' + '|'.join(negations_.keys()) + r')\b')


def is_year(text):
    if (len(text) == 3 or len(text) == 4) and (
            MIN_YEAR < len(text) < MAX_YEAR):
        return True
    else:
        return False


class TwitterPreprocessor:
    '''preprocessor for twitter data'''

    def __init__(self, text: str):
        self.text = text

    def fully_preprocess(self):
        return self \
            .remove_urls() \
            .remove_mentions() \
            .remove_hashtags() \
            .remove_twitter_reserved_words() \
            .remove_punctuation() \
            .remove_single_letter_words() \
            .remove_blank_spaces() \
            .remove_stopwords() \
            .remove_numbers()

    def remove_urls(self):
        self.text = re.sub(pattern=get_url_patern(), repl='', string=self.text)
        return self

    def remove_punctuation(self):
        self.text = self.text.translate(
            str.maketrans('', '', string.punctuation))
        return self

    def remove_mentions(self):
        self.text = re.sub(
            pattern=get_mentions_pattern(),
            repl='',
            string=self.text)
        return self

    def remove_hashtags(self):
        self.text = re.sub(
            pattern=get_hashtags_pattern(),
            repl='',
            string=self.text)
        return self

    def remove_twitter_reserved_words(self):
        self.text = re.sub(
            pattern=get_twitter_reserved_words_pattern(),
            repl='',
            string=self.text)
        return self

    def remove_single_letter_words(self):
        self.text = re.sub(
            pattern=get_single_letter_words_pattern(),
            repl='',
            string=self.text)
        return self

    def remove_blank_spaces(self):
        self.text = re.sub(
            pattern=get_blank_spaces_pattern(),
            repl=' ',
            string=self.text)
        return self

    def remove_stopwords(self, extra_stopwords=None):
        if extra_stopwords is None:
            extra_stopwords = []
        text = nltk.word_tokenize(self.text)
        stop_words = set(stopwords.words('english'))

        new_sentence = []
        for w in text:
            if w not in stop_words and w not in extra_stopwords:
                new_sentence.append(w)
        self.text = ' '.join(new_sentence)
        return self

    def remove_numbers(self, preserve_years=False):
        text_list = self.text.split(' ')
        for text in text_list:
            if text.isnumeric():
                if preserve_years:
                    if not is_year(text):
                        text_list.remove(text)
                else:
                    text_list.remove(text)

        self.text = ' '.join(text_list)
        return self

    def lowercase(self):
        self.text = self.text.lower()
        return self

    def handle_negations(self):
        self.text = re.sub(
            pattern=get_negations_pattern(),
            repl='',
            string=self.text)
        return self
