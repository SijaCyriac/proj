better_profanity and gensim.models word2vec - appropriate synonymns not found in training data
similar_words = word2vec_model.wv.most_similar(word, topn=10)

wordnet didn't have the harmful words

profanity_check and profanity_filter both are dependent on older version of scikit learn. Initial scikit-learn 1.2.2 version. We tried downgrading it to 0.20.2

engine=text-davinci-003 or text-davinci-002 (deprecated). model=gpt-3.5-turbo model is typically used with the ChatCompletion.create and engine=text-davinci-003 is used with completion.create. Completion but not supported in gpt>1 version. So tried using older version "pip install openai==0.28" with ChatCompletion and gpt 3.5 turbo but this is not free.

used GPT2Tokenizer with hugging face instead of gpt 3.5 turbo as it is unsuitable.

