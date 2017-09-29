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

# Globals.
data_root_path = "."
dataset_name = "intents-de"
intents = None

# Word vectors.
word_vectors_model_filename = "wiki.de.vec"
word_vectors_length = 300
word_vectors_model = None
#word_vectors_model_filename = "german.model"

# Preprocessing and training.
preprocess_data_anyway = False
overall_words = None
ignored_words = None
classe_names = None

model = None
train_model_anyway = False
train_x = None
train_y = None
num_epochs = 1000
batch_size = 8

context = ""

def main(args=None):
    process_arguments(args)
    preprocess_data()
    train_model()
    #evaluate_model()

def process_arguments(args):
    print("Implement!")

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
    preprocessed_data_path = os.path.join(data_root_path, preprocessed_data_root, dataset_name + "-word_embeddings.p")

    # Load the data from JSON-file.
    global intents
    dataset_path = os.path.join(data_root_path, dataset_name + ".json")
    with open(dataset_path) as json_data:
        intents = json.load(json_data)

    global words, classes, documents, train_x, train_y
    if os.path.exists(preprocessed_data_path) and preprocess_data_anyway == False:
        print("Preprocessed file already exists. Loading data...")
        raise Exception("Implement!")
        #words, classes, documents, train_x, train_y = pickle.load(open(preprocessed_data_path, "rb" ))
    else:
        print("Preparing training data...")
        load_word_vectors_lazily()

        with open(dataset_path) as json_data:

            intents = json.load(json_data)
            process_intents(intents)

            print("Overall words:", len(overall_words), overall_words)
            print("Ignored words:", len(ignored_words), ignored_words)

            print("Writing preprocessed file...")
            pickle.dump((class_names, train_x, train_y), open(preprocessed_data_path, "wb"))
            print("Done.", preprocessed_data_path)

def process_intents(intents):

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

    overall_words = sorted(list(set(overall_words)))
    ignored_words = sorted(list(set(ignored_words)))

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
    word_vector = None
    try:
        # Try the original word.
        word_vector = word_vectors_model.wv[word]
    except KeyError:
        # Maybe there is a vector for the lower-case word.
        try:
            word_vector = word_vectors_model.wv[word.lower()]
        except KeyError as key_error:
            raise key_error
    return word_vector

def OLD_preprocess_data(): # TODO remove
    preprocessed_data_root = "preprocessed"
    if not os.path.exists(preprocessed_data_root):
        os.makedirs(preprocessed_data_root)
    preprocessed_data_path = os.path.join(data_root_path, preprocessed_data_root, dataset_name + "-word_embeddings.p")

    # Load the data from JSON-file.
    global intents
    dataset_path = os.path.join(data_root_path, dataset_name + ".json")
    with open(dataset_path) as json_data:
        intents = json.load(json_data)

    global words, classes, documents, train_x, train_y
    if os.path.exists(preprocessed_data_path) and preprocess_data_anyway == False:
        print("Preprocessed file already exists. Loading data...")
        words, classes, documents, train_x, train_y = pickle.load(open(preprocessed_data_path, "rb" ))
    else:
        print("Preparing training data...")

        # Extract data from the intents.
        words, classes, documents = process_intents(intents)
        print (len(documents), "documents")
        print (len(classes), "classes", classes)
        print (len(words), "unique stemmed words", words)

        # Create training-sets.
        train_x, train_y = create_training_data(words, classes, documents)

        # Save all.
        pickle.dump((words, classes, documents, train_x, train_y), open(preprocessed_data_path, "wb"))
        print("Training data saved.")

    print_training_data(train_x, train_y)


def OLD_process_intents(intents): # TODO remove
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    # remove duplicates
    classes = sorted(list(set(classes)))

    return words, classes, documents

def OLD_create_training_data(words, classes, documents): # TODO remove
    # create our training data
    training = []
    output = []
    # create an empty array for our output
    output_empty = [0] * len(classes)

    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # stem each word
        pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
        # create our bag of words array
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

        print(doc)
        print(classes[classes.index(doc[1])])

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)

    # create train and test lists
    train_x = list(training[:,0])
    train_y = list(training[:,1])

    return train_x, train_y


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
    model_filename = "{}-{}epochs-word_embeddings".format(dataset_name, num_epochs)
    model_path = os.path.join(data_root_path, model_root, model_filename)

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
