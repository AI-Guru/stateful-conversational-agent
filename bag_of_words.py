import nltk
import numpy as np

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

sentence = """Zwar bin ich gescheiter als all die Laffen,
Doktoren, Magister, Schreiber und Pfaffen;
Mich plagen keine Skrupel noch Zweifel,
Fürchte mich weder vor Hölle noch Teufel –
Dafür ist mir auch alle Freud entrissen,
Bilde mir nicht ein, was Rechts zu wissen,
Bilde mir nicht ein, ich könnte was lehren,
Die Menschen zu bessern und zu bekehren."""

def main():
    vocabulary = get_vocabulary_from_corpus(corpus)
    print("Vocabulary length:", len(vocabulary))
    print("Vocabulary:", vocabulary)

    bag_of_words = get_bag_of_words(sentence, vocabulary)
    print("Bag-of-words:", bag_of_words)

def get_vocabulary_from_corpus(corpus):
    tokens = get_tokens_from_text(corpus)
    vocabulary = sorted(list(set(tokens)))
    return np.array(vocabulary)

def get_bag_of_words(sentence, vocabulary):
    tokens = get_tokens_from_text(sentence)
    bag = [0] * len(vocabulary)
    for token in tokens:
        for index, vocable in enumerate(vocabulary):
            if token == vocable:
                bag[index] = 1

    return np.array(bag)

def get_tokens_from_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    return tokens

if __name__ == "__main__":
    main()
