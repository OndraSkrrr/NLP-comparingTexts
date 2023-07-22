from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
from scipy import spatial
from fpdf import FPDF
from tika import parser
import math
import os
import requests
import bs4
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

def jaccard_sim(A, B):
    return len(A.intersection(B))/len(A.union(B))
"""
textA = "I am frying pancakes for dinner in my new kitchen tonight." 
textB = "I fried pancakes for dinner in my new kitchen yesterday."
Toto som ulozil do textovych suborov v zlozke databaza
Pri volani funkcie LoadingText treba zadavat cely nazov suboru, cize Text1.txt a Text2.txt
"""
def LoadText():
    text= input("Enter the .txt file name:")
    f = open(f'Databaza/{text}.txt','r')
    obsah = f.read()
    return(obsah)

def LoadPDFText():
    text= input("Enter the .pdf file name:")
    raw = parser.from_file(f'Databaza/{text}.pdf')
    obsah = raw['content']
    return(obsah)

def urlToText():
    adresa = input("Zadejte URL stránky: ")
    page = requests.get(adresa)
    data = bs4.BeautifulSoup(page.content, "html.parser")
    return(data.getText())

Input = -1
while(not Input in ['1', '2', '3']):
    Input = input('What kind of file do u want to compare?\n'
                     '.txt file ==> [1]\n'
                     '.pdf file ==> [2]\n'
                     'from URL  ==> [3]\n'
                     'Input:')

if Input == '1':
    textA = LoadText()
    textB = LoadText()
elif Input == '2':
    textA = LoadPDFText()
    textB = LoadPDFText()
elif Input == '3':
    textA = urlToText()
    textB = urlToText()
else:
    print("Invalid choice!")


listA = textA.split()
listB = textB.split()
#uniquelist = set(listA).union(set(listB))

textAL = ""
listAL = []
for word in listA:
    word = word.replace('.', '')
    word = word.replace(',', '')
    textAL += " " + wnl.lemmatize(word)
    listAL.append(wnl.lemmatize(word))
textBL = ""
listBL = []
for word in listB:
    word = word.replace('.', '')
    word = word.replace(',', '')
    textBL += " " + wnl.lemmatize(word)
    listBL.append(wnl.lemmatize(word))


##print("\n")
##print(textAL)
##print(textBL)
##print(listBL)
##print(listAL)

vectorizer = CountVectorizer()
vectorizer.fit([textAL, textBL])
##print(vectorizer.vocabulary_)
vectorA = vectorizer.transform([textAL])
vectorB = vectorizer.transform([textBL])

##print(vectorA.toarray())
vect_a=vectorA.toarray()
##print(vectorB.toarray())
vect_b=vectorB.toarray()

dist_euc = distance.euclidean(vect_a, vect_b)

jaccard_val = jaccard_sim(set(listAL), set(listBL))

vectorizer2 = TfidfVectorizer()
vectors = vectorizer2.fit_transform([textA,textB])
feature_names = vectorizer2.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
vect1,vect2= denselist
cosine_distance = 1 - spatial.distance.cosine(vect1, vect2)
cosine_sim = "{:.0%}".format(cosine_distance)

##Hledani nejdulezitejsich slov
korpus = vectorizer2.vocabulary_
dulezite = []
#print(korpus)

hranice = sorted(vect1)[-5]
for i in range(0,len(vect1)):
    if vect1[i] > hranice:
        for a in korpus.keys():
            if korpus[a] == i:
                dulezite.append(a)

hranice = sorted(vect2)[-5]
for i in range(0,len(vect2)):
    if vect2[i] > hranice:
        for a in korpus.keys():
            if korpus[a] == i:
                dulezite.append(a)
dulezite = set(dulezite)
setToStr = ', '.join(map(str, dulezite))
#print(setToStr)



pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', size=15)

pdf.cell(200, 15,txt = f"Eucleidovska vzdalenost textu je {dist_euc} ",ln=1,align='L')
pdf.cell(200, 15,txt = f"Cosine Distance is {cosine_distance}",ln = 2,align = 'L')
pdf.cell(200, 15,txt = f"Texty obsahuji z {math.trunc(jaccard_val*100)} % stejna slova",ln=3,align='L')
pdf.cell(200, 15,txt = "Slova s nejvetsim TF-IDF: ",ln=3,align='L')
pdf.cell(200, 15,txt = setToStr,ln=3,align='L')

if dist_euc == 0 and cosine_distance == 1:
    pdf.set_font('Arial','B', size=20)
    pdf.cell(200, 15, txt = 'Jedna sa o rovnake texty',ln = 4,align='L')
pdf.output('Output.pdf')

os.system(r'Output.pdf')
print('Subor Output.pdf sa otvoril')



