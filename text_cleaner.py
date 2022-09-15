import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

class TextCleaner:
    '''TextCleaner for text preprocessing'''
    def __init__(self):
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.punctuations = list(string.punctuation)
        self.stopwords = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()

    def lower_casing(self, x):
        # convert string to lower case
        return x.lower()

    def remove_punctuations_stopwords(self, tokenized_x):
        # remove punctuations and stopwords
        tokenized_x = [word for word in tokenized_x if (word not in self.punctuations) and (word not in self.stopwords)]
        return tokenized_x

    def word_lemmatizing(self, tokenized_x):
        # word lemmatizing: converts the word into its root word, Example: reduce words such as “am”, “are”, and “is” to a common form such as “be”
        tokenized_x = [self.lemmatizer.lemmatize(word) for word in tokenized_x]
        return tokenized_x

    def clean_text(self, x):
        # clean text
        x = x.strip()
        x = self.lower_casing(x)
        tokenized_x = word_tokenize(x)
        tokenized_x = self.remove_punctuations_stopwords(tokenized_x)
        tokenized_x = self.word_lemmatizing(tokenized_x)
        return " ".join(tokenized_x)