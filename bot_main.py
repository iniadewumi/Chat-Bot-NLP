import json, pickle, nltk, pathlib, random
import numpy as np
from tensorflow.keras import models

HOME = pathlib.Path().resolve()
MODELS_DIR = HOME/"MODELS"
INTENTS_DIR = HOME/"INTENTS"


class ChatBot:
    def __init__(self):
        self.model = models.load_model(MODELS_DIR/"chatbot_model.h5")
        with open('DATASETS/LABELS.pickle', 'rb') as f1, open('DATASETS/words.pickle', 'rb') as f2, open(INTENTS_DIR/'intents.json', 'r') as f3:
            self.LABELS = pickle.load(f1)
            self.WORDS = pickle.load(f2)
            self.INTENTS = json.load(f3)
            self.END = False
            self.stem = nltk.stem.lancaster.LancasterStemmer()
    def preprocess_sentence(self, sentence):
        return [self.stem.stem(word) for word in nltk.word_tokenize(sentence)]
        
    def bag_of_words(self, sentence):
        sent_words = self.preprocess_sentence(sentence)   
        bag = [0]*len(self.WORDS)
        for word in sent_words:
            if word in self.WORDS:
                bag[self.WORDS.index(word)] = 1
        return np.array(bag)                
    
    def start(self, sentence):
        ERR_THRESH = 0.40
        bag = self.bag_of_words(sentence)
        results = self.model.predict(np.array([bag]))[0]
        print(results)
        result = [[i, r] for i,r in enumerate(results) if r>ERR_THRESH]
        if not result:
            return "I did not understand that, sorry!"
        result.sort(key=lambda x: x[1], reverse=True)

        self.curr_result = [{"Intent": self.LABELS[r[0]], "Probability": r[1]} for r in result]
        return self.response(self.curr_result)
    
    
    def response(self, intents):
        main_tag = intents[0]["Intent"]
        if main_tag == 'goodbye':
            self.END = True
        main_intent = next(x for x in self.INTENTS['intents'] if x['tag']==main_tag)   
        return random.choice(main_intent['responses'])
    
    
    
if __name__ == '__main__':
    bot = ChatBot()
    while not bot.END:
        sentence = input("\nEnter Chat: ")
        response = bot.start(sentence)
        print(response)