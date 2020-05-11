

import numpy as np
import re

from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle 


from flask import Flask
from flask import render_template, request

def predict(news):
    file_model = 'saved_model_fake_news.sav'
    file_tfidf = 'saved_tfidf.pickle'

    tfidf = pickle.load(open(file_tfidf,'rb'))
    saved_clf = pickle.load(open(file_model,'rb'))

    processed = tfidf.transform(news)
    result = saved_clf.predict(processed)
    return result

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
            ' '.join(emoticons).replace('-', '')
    return text

def tokenizer_stemmer(text):
    porter = PorterStemmer()
    stop = stopwords.words('english')
    temp = [porter.stem(word) for word in text.split()]
    return [w for w in temp if w not in stop]



app = Flask(__name__)


@app.route('/')

def index():
	return render_template('index.html')
@app.route('/', methods=['POST'])

def get_news():
	news = request.form['news']
	result = predict(np.array([news]))
	if(result==0):
		result='valid. You can rely on this.'
	elif(result==1):
		result='not valid. Do not rely on this '
	return render_template('result.html',result= result)
app.run(debug=True) 





