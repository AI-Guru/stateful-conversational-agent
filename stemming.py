import nltk
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import GermanStemmer
stemmer = GermanStemmer()

corpus = """Habe nun, ach! Philosophie,
Juristerei und Medizin,
Und leider auch Theologie
Durchaus studiert, mit heißem Bemühn.
Da steh ich nun, ich armer Tor!
Und bin so klug als wie zuvor;
Heiße Magister, heiße Doktor gar
Und ziehe schon an die zehen Jahr
Herauf, herab und quer und krumm
Meine Schüler an der Nase herum –
Und sehe, daß wir nichts wissen können!
Das will mir schier das Herz verbrennen."""

def main():
    vocabulary = get_vocabulary_from_corpus(corpus, use_stemmer=False)
    print("Vocabulary length:", len(vocabulary))
    print("Vocabulary:", vocabulary)

    stemmed_vocabulary = get_vocabulary_from_corpus(corpus, use_stemmer=True)
    print("Stemmed vocabulary length:", len(stemmed_vocabulary))
    print("Stemmed vocabulary:", stemmed_vocabulary)

def get_vocabulary_from_corpus(corpus, use_stemmer):
    tokens = get_tokens_from_text(corpus)
    if use_stemmer is True:
        tokens = [stemmer.stem(token) for token in tokens]
    vocabulary = sorted(list(set(tokens)))
    return np.array(vocabulary)

def get_tokens_from_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    return tokens

if __name__ == "__main__":
    main()
