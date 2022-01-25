import pandas
pandas.options.mode.chained_assignment = None
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import sys

raw_data = pandas.read_csv(sys.argv[1])
expandable_words = pandas.read_csv('expansion.csv')

wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))
STOPWORDS = set(" ".join(STOPWORDS).replace("'","").split())
# STOPWORDS.add('feel')

def preprocess(text):
    words = text.split()
    
    expanded_text = []
    for word in words:
        expandable = False
        for i in range(len(expandable_words['word'])):
            if expandable_words['word'][i] == word:
                expandable = True
                expanded_text.append(expandable_words['expandedWord'][i])
                break
        if not expandable:
            expanded_text.append(word)
    expanded_text = (" ".join(expanded_text)).split()
    
    pos_tagged_text = nltk.pos_tag(expanded_text)
    lemmatized_text = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text]

    final_text = [word for word in lemmatized_text if word not in STOPWORDS]
    return " ".join(final_text)


for i in range(len(raw_data['text'])):
    raw_data['text'][i] = preprocess(raw_data['text'][i])
raw_data.to_csv(sys.argv[2], index=False)
