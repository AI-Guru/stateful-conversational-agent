import os
import pickle
import json
import nltk
import random
import nltk
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import keras
from keras.models import load_model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Masking
from keras.layers import MaxPooling1D, Conv1D, Conv2D
from keras.layers import Activation, MaxPooling2D, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


class Variant:
    WORD_VECTORS_SUM = "word-vectors-sum" # Sums up all word vectors.
    WORD_VECTORS_SEQUENCE = "word-vectors-list" # Generates a list from all word vectors.
    WORD_VECTORS_SEQUENCE_LSTM = "word-vectors-sequence-lstm" # Generates a list from all word vectors for Seq2One.

variant = Variant.WORD_VECTORS_SUM

# Globals.
data_root_path = "."
dataset_name = "intents-de"
intents = None

# Word vectors.
word_vectors_model_filename = "wiki.de.vec"
word_vectors_model_filename = "german.model"
word_vectors_length = 300
word_vectors_model = None

# Preprocessing and training.
preprocess_data_anyway = True
overall_words = None
ignored_words = None
classe_names = None

model = None
train_model_anyway = False
train_x = None
train_y = None
num_epochs = 1000
batch_size = 8

if preprocess_data_anyway == True:
    print("WARNING! Data will be preprocessed anyway!")
if train_model_anyway == True:
    print("WARNING! Model will be trained anyway!")

# Validation.
raise Exception("Load this properly from file!")
validate_data_raw = [
    "Ist ihr Laden heute offen?"
    "Nehmen Sie Bargeld?",
    "Welche Sorten von Mopeds vermieten Sie?",
    "Tschüss, auf wiedersehen.",
    "Wir möchten ein Moped mieten.",
    "Heute."]
validate_data = None

# For the finite state machine.
context = ""

# TODOs
# TODO Shuffle training-data?
# TODO process test data too

def main(args=None):
    process_arguments(args)
    preprocess_data()
    train_model()
    #evaluate_model()

# TODO remove this?
def process_arguments(args):
    print("Implement!")


# TODO move this down
def load_word_vectors_lazily():
    global word_vectors_model
    if word_vectors_model != None:
        return
    word_vectors_model_root = "embeddings"
    word_vectors_model_path = os.path.join(word_vectors_model_root, word_vectors_model_filename)
    print("Loading word vectors from",  word_vectors_model_path + "...")
    word_vectors_model_size = os.path.getsize(word_vectors_model_path) / (1024.0 * 1024.0)
    print("File size is {0:.2f}MB".format(word_vectors_model_size))
    print("Be patient! This might take a while...")
    try:
        print("Attempting to load as vector-format...")
        word_vectors_model = KeyedVectors.load_word2vec_format(word_vectors_model_path, binary=False)
        print("Success!")
    except:
        print("Failed!")
        try:
            print("Attempting to load as binary-format...")
            word_vectors_model = KeyedVectors.load_word2vec_format(word_vectors_model_path, binary=True)
        except:
            print("Failed!")
            exit(0)

    print("Success!")

def preprocess_data():
    preprocessed_data_root = "preprocessed"
    if not os.path.exists(preprocessed_data_root):
        os.makedirs(preprocessed_data_root)
    preprocessed_data_filename = "{}-{}.p".format(dataset_name, word_vectors_model_filename)
    preprocessed_data_path = os.path.join(data_root_path, preprocessed_data_root, preprocessed_data_filename)
    print("PREPROCESSED DATA PATH:", preprocessed_data_path)

    global class_names, train_x, train_y, validate_data
    if os.path.exists(preprocessed_data_path) and preprocess_data_anyway == False:
        print("Preprocessed file already exists. Loading data...")
        intents, class_names, train_x, train_y, validate_data = pickle.load(open(preprocessed_data_path, "rb" ))
    else:
        print("Preparing training data...")
        load_word_vectors_lazily()

        # Load the data from JSON-file.
        global intents
        dataset_path = os.path.join(data_root_path, dataset_name + ".json")
        with open(dataset_path) as json_data:
            intents = json.load(json_data)

            process_intents()

            process_validate_data()

            global overall_words
            global ignored_words
            overall_words = sorted(list(set(overall_words)))
            ignored_words = sorted(list(set(ignored_words)))
            print("Overall words:", len(overall_words), overall_words)
            print("Ignored words:", len(ignored_words), ignored_words)

            print("Writing preprocessed file...")
            pickle.dump((intents, class_names, train_x, train_y, validate_data), open(preprocessed_data_path, "wb"))
            print("Done.", preprocessed_data_path)

def process_intents():

    print("Processing intents to generate training data...")

    global intents
    global class_names
    class_names = []
    for intent in intents["intents"]:
        tag = intent["tag"]
        class_names.append(tag)
    class_names = sorted(list(set(class_names)))
    print("Class names: ", class_names)

    global overall_words
    overall_words = []
    global ignored_words
    ignored_words = []
    global train_x
    train_x = []
    global train_y
    train_y = []
    # TODO progress bar!
    for intent in intents["intents"]:
        tag = intent["tag"]
        one_hot_vector = get_one_hot_vector_from_word(tag, class_names)

        for pattern in intent["patterns"]:
            tokens = get_tokens_from_text(pattern)
            word_vectors, ignored = get_word_vectors_from_text(pattern)
            overall_words.extend(tokens)
            ignored_words.extend(ignored)
            train_x.append(word_vectors)
            train_y.append(one_hot_vector)

    print("Done.")

def process_validate_data():

    print("Processing data for validation...")

    global validate_data
    validate_data = []
    for validate_item_raw in validate_data_raw:
        tokens = get_tokens_from_text(validate_item_raw)
        word_vectors, ignored = get_word_vectors_from_text(validate_item_raw)
        overall_words.extend(tokens)
        ignored_words.extend(ignored)
        validate_data.append(word_vectors)

    print("Done.")

def get_one_hot_vector_from_word(word, class_names):
    one_hot_vector = []
    for class_name in class_names:
        if class_name is word:
            one_hot_vector.append(1)
        else:
            one_hot_vector.append(0)
    return np.array(one_hot_vector)

def get_word_vectors_from_text(text):
    tokens = get_tokens_from_text(text)
    word_vectors = []
    ignored = []
    for token in tokens:
        try:
            word_vector = get_word_vector(token)
            word_vectors.append(word_vector)
        except KeyError:
            ignored.append(token)
    return np.array(word_vectors), ignored

def get_tokens_from_text(text, to_lower=False):
    tokens = nltk.word_tokenize(text)
    if to_lower == True:
        tokens = [token.lower() for token in tokens]
    return tokens

def get_word_vector(word):

    # Try the original word.
    try:
        word_to_use =  word
        word_vector = word_vectors_model.wv[word_to_use]
        return word_vector
    except KeyError:
        pass

    # Try the original word with replaced umlauts.
    try:
        word_to_use =  replace_special_characters(word)
        word_vector = word_vectors_model.wv[word_to_use]
        return word_vector
    except KeyError:
        pass

    # Try the original word in lower case.
    try:
        word_to_use =  word.lower()
        word_vector = word_vectors_model.wv[word_to_use]
        return word_vector
    except KeyError:
        pass

    # Try the original word in lower case and replaced umlauts.
    try:
        word_to_use =  replace_special_characters(word.lower())
        word_vector = word_vectors_model.wv[word_to_use]
        return word_vector
    except KeyError:
        pass

    exception_message = "No word vector found for " + word
    print(exception_message)
    raise KeyError(exception_message)


def replace_special_characters(word):
    word = word.replace("ä", "ae")
    word = word.replace("ö", "oe")
    word = word.replace("ü", "ue")
    word = word.replace("Ä", "Ae")
    word = word.replace("Ö", "Oe")
    word = word.replace("Ü", "Ue")
    word = word.replace("ß", "ss")
    return word


# TODO Does this still work?
def print_training_data(train_x, train_y):
    print(len(train_x), "train_x")
    print(len(train_y), "train_y")

    for i in range(len(train_x)):
        print(str(i) + ":", "".join(str(t) for t in train_x[i]), "->", "".join(str(t) for t in train_y[i]))
        #print(documents[i][0])
        #print(intents["intents"][i]["patterns"])
        #print(intents["intents"][i]["responses"])

def train_model():
    if len(train_x) != len(train_y):
        raise Exception("Training data inconsistent!")

    model_root = "model"
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    model_filename = "{}-{}-{}-{}epochs".format(dataset_name, word_vectors_model_filename, variant, num_epochs)
    model_path = os.path.join(data_root_path, model_root, model_filename)
    print("MODEL PATH:", model_path)

    global model
    if os.path.exists(model_path) and train_model_anyway == False:
        print("Model already exists. Loading model...")
        model = load_model(model_path)
    else:
        print("Training model...")

        tensorBoard = keras.callbacks.TensorBoard(log_dir='./logs',
                                                  histogram_freq=1,
                                                  write_graph=True,
                                                  write_images=True)

        # TODO process training data wrt variant

        # TODO create model wrt variant

        # TODO train

        raise Exception("Implement properly!")

        # Some hyperparameters.
        input_length = len(train_x[0])
        num_classes = len(train_y[0])

        # Model architecture.
        model = Sequential()
        model.add(Dense(input_length, input_shape=(input_length,)))
        model.add(Dense(8))
        model.add(Dense(8))
        model.add(Dense(num_classes, activation="softmax"))

        # For a mean squared error regression problem.
        model.compile(optimizer='rmsprop', loss='mse')

        # Training.
        model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, callbacks=[tensorBoard])

        # Saving the model.
        print("Saving model...")
        model.save(model_path)

        print("Model saved.")

# TODO rewrite this
def evaluate_model():

    texts = []
    texts.append("is your shop open today?")
    texts.append("do you take cash?")
    texts.append("what kind of mopeds do you rent?")
    texts.append("Goodbye, see you later")
    texts.append("we want to rent a moped")
    texts.append("today")

    text = ""
    text = "is your shop open today?"

    if len(texts) == 0:
        text = input("Please enter something: ")
        texts.append(text)

    for text in texts:
        evaluate_text(text)


def evaluate_text(text):
    # Generate the bag of words for the input text.
    bag_of_words = get_bag_of_words(text, words)

    # Predict using the Neural Net.
    bag_of_words = np.expand_dims(bag_of_words, axis=0)
    prediction = model.predict(bag_of_words)[0]

    # Process the results.
    prediction_maximum = get_prediction_maximum(prediction)
    prediction_response, new_context = get_prediction_response(prediction_maximum)

    print("Context:    ", context)
    print("Text:       ", text)
    print("Prediction: ", prediction_maximum)
    print("Response:   ", prediction_response)
    print("New context:", new_context)

    global context
    context = new_context


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def get_bag_of_words(sentence, vocabulary):
    words = nltk.word_tokenize(sentence)
    words = [word.lower() for word in words]
    words = [stemmer.stem(word) for word in words]
    bag = [0] * len(vocabulary)
    for word in words:
        for index, vocable in enumerate(vocabulary):
            if word == vocable:
                bag[index] = 1

    return np.array(bag)

def prediction_to_string(prediction):
    prediction_map = {}
    for i in range(len(prediction)):
        p = prediction[i]
        class_name = classes[i]
        prediction_map[class_name] = p
    return str(prediction_map)

def get_prediction_maximum(prediction):
    index = np.argmax(prediction)
    return classes[index]

def get_prediction_response(class_name):
    response = "Sorry, I have no answer."
    new_context = ""

    for intent in intents['intents']:
        # find a tag matching the first result
        if intent['tag'] == class_name:

            if context == "" or ("context_filter" in intent and context == intent['context_filter']):
                response = random.choice(intent['responses'])
                if "context_set" in intent:
                    new_context = intent['context_set']

    return response, new_context

if __name__ == "__main__":
    main()
