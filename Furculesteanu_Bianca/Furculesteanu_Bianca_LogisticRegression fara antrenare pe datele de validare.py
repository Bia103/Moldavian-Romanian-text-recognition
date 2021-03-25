!pip install stop_words
import numpy as np
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
from stop_words import get_stop_words
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
stop_words = get_stop_words('ro')

with open('train_samples.txt', mode="r", encoding="utf-8") as f:
     tr_sa = f.readlines()

train_samples = []

for linie in tr_sa:
     id, prop = linie.split('\t')
     prop = prop.rstrip()
     train_samples.append(prop)
     


train_labels = np.loadtxt('train_labels.txt', 'int')
#test_labels = np.loadtxt('validation_source_labels.txt', 'int')

test_samples = []
test_lab = []

with open('test_samples.txt', mode="r", encoding="utf-8") as f:
     te_sa = f.readlines()

for linie in te_sa:
     id, prop = linie.split('\t')
     prop = prop.rstrip()
     test_samples.append(prop)
     test_lab.append(id)

dict_tr = {}
dict_v = []
i = 0
j = 0
print(len(tr_sa))
for sent in train_samples:
  aux = sent.split()
  for w in aux:
    if w not in dict_tr:
      dict_tr[w] = i
      i = i + 1
      dict_v.append(w)
print(len(dict_tr))
print("Doamne ajuta")


result_tr = np.zeros((len(train_samples), len(dict_tr)))
print("Doamne ajuta nr2")



i = 0
for sent in train_samples:
    aux = sent.split()
    for w in aux:
      result_tr[i][dict_tr[w]] += 1
    i = i + 1
print("Si aici 1")

result_te = np.zeros((len(test_samples), len(dict_tr)))
i = 0
for sent in test_samples:
    aux = sent.split()
    for w in aux:
      if w in dict_v:
        result_te[i][dict_tr[w]] += 1 
    i = i + 1

scaler = preprocessing.StandardScaler()
print(result_tr)

scaler.fit(result_tr)

result_tr = scaler.transform(result_tr)

result_te = scaler.transform(result_te)

clf = LogisticRegression()
#clf = SVC(C = 100, kernel='linear', probability = True)
# probabil mai tarziu am sa am o eroare aici

tr_l = []
for labels in train_labels:
    tr_l.append(labels[1])
#te_l = []
#for labels in test_labels:
#    te_l.append(labels[1])
print("ajunge2")
#print("hop sarmaua")
print(tr_l[1])
clf.fit(result_tr, tr_l)
print("te rog sa ajungi aici")
prediction = clf.predict(result_te)
print(clf.decision_function(result_te))
#clf.predict_proba(result_te)
clf.predict(result_te)
#print()
print(prediction.shape)
#print(te_l)
#for i in range(len(prediction)):
 #     prediction[i] = int(prediction[i])
#for i in range(len(te_l)):
#      te_l[i] = int(te_l[i])
#print(np.mean(predict == te_l))
#print(len(predict))
#print(len(te_l))
submission = pd.DataFrame({'id':test_lab,'label':prediction})
submission.head()
#print(result_te.shape)
#print(result_te.shape)
filename = 'sample_submission.csv'

submission.to_csv(filename,index=False)
np.savetxt("file.txt",prediction)
print('Saved file: ' + filename)
print(tr_l)
