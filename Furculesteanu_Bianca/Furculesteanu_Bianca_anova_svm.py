
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


with open('train_samples.txt', mode="r", encoding="utf-8") as f:
     tr_sa = f.readlines()                                          # citirea datelor din training samples

train_samples = []

for linie in tr_sa:                                                 # spargerea in propozitii a datelor citite din train_samples.txt
     id, prop = linie.split('\t')
     prop = prop.rstrip()
     train_samples.append(prop)
     
with open('validation_samples.txt', mode="r", encoding="utf-8") as f:   
     va_sa = f.readlines()                                          # citirea datelor din validation samples

validation_samples = []

for linie in va_sa:                                                 # spargerea in propozitii a datelor citite din validation_samples.txt
     id, prop = linie.split('\t')
     prop = prop.rstrip()
     validation_samples.append(prop)

train_labels = np.loadtxt('train_labels.txt', 'int')                # citirea label-urilor din train_labels.txt
validation_labels = np.loadtxt('validation_labels.txt', 'int')      # citirea label-urilor din validation_labels.txt

test_samples = []
test_lab = []

with open('test_samples.txt', mode="r", encoding="utf-8") as f:
     te_sa = f.readlines()                                          # citirea datelor din test_samples.txt

for linie in te_sa:                                                 # spargerea in propozitii a datelor citite din test_samples.txt
     id, prop = linie.split('\t')
     prop = prop.rstrip()
     test_samples.append(prop)
     test_lab.append(id)

dict_tr = {}    #dictionarul in care am pastrat id-ul pentru fiecare cuvant
dict_v = []     # vector ce contine fircare cuvant ce se afla in dictionar
i = 0           #id-ul care va fi atribuit cuvantului 
j = 0
print(len(tr_sa))
for sent in train_samples:  #parcurgerea propozitiilor din train samples
  aux = sent.split()        # spargerea propozitiilor in cuvinte
  for w in aux:             # parcurgerea fiecarui cuvant
    if w not in dict_tr:    # daca cuvantul nu se afla in dictionar i 
      dict_tr[w] = i        # va atribui un id 
      i = i + 1             # incrementarea id-ului 
      dict_v.append(w)      # se adauga cuvantul in vectorul de cuvinte
for sent in validation_samples: # se aplica acelas procedeu pe datele de validare
  aux = sent.split()
  for w in aux:
    if w not in dict_tr:
      dict_tr[w] = i
      i = i + 1
      dict_v.append(w)

result_tr = np.zeros((len(train_samples)+len(validation_samples), len(dict_tr))) # se initializeaza matricea de features

i = 0                           # indicele propozitiei
for sent in train_samples:      # parcurgere
    aux = sent.split()          # impartim propozitiile in cuvinte
    for w in aux:
      result_tr[i][dict_tr[w]] += 1 # se numara de cate ori exista un cuvant intr-o propozitie
    i = i + 1                       # se trece la urmatoarea propozitie
for sent in validation_samples:     # se adauga si datele de validare in matrice
    aux = sent.split()
    for w in aux:
      result_tr[i][dict_tr[w]] += 1
    i = i + 1

result_te = np.zeros((len(test_samples), len(dict_tr)))
i = 0
for sent in test_samples:   # se aplica acelas procedeu pe datele de test
    aux = sent.split()
    for w in aux:
      if w in dict_v:
        result_te[i][dict_tr[w]] += 1 
    i = i + 1

scaler = preprocessing.StandardScaler()         #standardizarea datelor
scaler.fit(result_tr)
result_tr = scaler.transform(result_tr)
result_te = scaler.transform(result_te)

anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])

anova_svm.set_params(anova__k=10, svc__C=.1).fit(result_tr, tr_l)

tr_l = []                                       # se retin label-urile din training_labels si validation_labels intr-un vector
for labels in train_labels:
    tr_l.append(labels[1])
for labels in validation_labels:
    tr_l.append(labels[1])

print(tr_l[1])
clf.fit(result_tr, tr_l)
prediction = clf.predict(result_te)             # se fac predictiile
#print(clf.decision_function(result_te))
#clf.predict_proba(result_te)
clf.predict(result_te)

print(prediction.shape)

submission = pd.DataFrame({'id':test_lab,'label':prediction})
submission.head()
filename = 'sample_submission.csv'

submission.to_csv(filename,index=False)
np.savetxt("file.txt",prediction)
print('Saved file: ' + filename)
#print(tr_l)
