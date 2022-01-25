import json, pickle, nltk, pathlib
import numpy as np


stem = nltk.stem.lancaster.LancasterStemmer()



HOME = pathlib.Path().resolve()
INTENTS_DIR = HOME/"INTENTS"
DATASETS = HOME/"DATASETS"
WORDS, DOCS  = [], []


with open(INTENTS_DIR / "intents_first.json") as f:
    intents = json.load(f)

LABELS = [x['tag'] for x in intents['intents']]

for intent in intents["intents"]:
    WORDS.extend([nltk.tokenize.word_tokenize(pattern) for pattern in intent['patterns']])
    DOCS.append([(nltk.tokenize.word_tokenize(pattern), intent['tag']) for pattern in intent['patterns']])

WORDS = [item for sublist in WORDS for item in sublist if item.isalpha()]
DOCS = [item for sublist in DOCS for item in sublist]

WORDS = sorted(
    {stem.stem(word) for word in WORDS}
)

with open(DATASETS/'LABELS.pickle', 'wb') as f1, open(DATASETS/'words.pickle', 'wb') as f2:
    pickle.dump(LABELS, f1)
    pickle.dump(WORDS, f2)

output, training = [], []
out_empty = [0 for _ in range(len(LABELS))]


for doc in DOCS:
    bag = []

    word_patterns = [stem.stem(w) for w in doc[0]]

    for w in WORDS:
        if w in word_patterns:
            bag.append(1)
        else:
            bag.append(0)

    output_row = list(out_empty)
    output_row[LABELS.index(doc[1])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

data = {
    "training": training,
    "output": output
}

with open(DATASETS/"training.pickle", 'wb') as f:
    pickle.dump(data, f)