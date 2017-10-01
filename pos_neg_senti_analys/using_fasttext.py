import fasttext
model = fasttext.load_model('model.bin')
texts = ['This is not sarcasm', 'Im Serious!']
labels = classifier.predict(texts)
print (labels)