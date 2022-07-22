import csv
f = open("train_dataset_train.csv", encoding='utf-8')
read_oblivian = csv.reader(f, delimiter = ",")
data = []
new_Category=[]
categories = set()
m=1
alf="а бвгдеёжзийклмнопрстуфхцчшщъыьэюя"
def clean(txt):
    returnus_mobius=""
    for t in txt.lower():
        if not((" " == ("0"+returnus_mobius)[-1]) and (" " == t)):
            if(t in alf):
                returnus_mobius+=t
    return returnus_mobius


for line in read_oblivian:
    ID, text, topic, person, category = line
    if(m==1):
        m=100
    else:
        text=clean(text)
        topic=clean(topic)
        person=clean(person)
        data.append([int(ID), text, topic, person, int(category)])
    categories.add(category)


categories=list(categories)
for c in categories:
    if(c.isdigit()):
        new_Category.append(int(c))
categories=new_Category
del new_Category
(categories).sort()
print(categories)
New_data=[]
met=0
textes=""
for i in range(len(data)):
    met=data[i][4]
    textes=data[i][1]+data[i][2]+data[i][3]
    New_data.append([met,textes])
data=New_data.copy()
del New_data
#print(data)
# NLTK - библиотек Python для решения задач обработки естественного языка
import nltk
from nltk.stem.snowball import SnowballStemmer    # Стеммер Porter2 - новая версия стеммера Портера
from nltk.corpus import stopwords                 # Библиотека стоп-слов
from nltk import word_tokenize                    # Токенизатор
stemmer = SnowballStemmer("russian")              # Стеммер для русского языка
russian_stopwords = stopwords.words("russian")    # Список стоп-слов для русского языка
# Расширение списка стоп-слов (см. набор данных)
russian_stopwords.extend(['…', '«', '»', '...', 'т.д.', 'т', 'д', 'nan',"г"])
# Получение нулевой формы слов
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
# Провердим лемматизацию и избавляемся от стоп-слов
lemm_texts_column = []
for u in range(len(data)):
    tokens = word_tokenize(data[u][1])
    lemmed_tokens = [morph.parse(token)[0].normal_form for token in tokens if token not in russian_stopwords]
    data[u][1] = " ".join(lemmed_tokens)
    lemm_texts_column.append(data[u][1])


import pandas as pd
# CountVectorizer - класс конвертации текста в матрицу токенов
from sklearn.feature_extraction.text import CountVectorizer
df = pd.DataFrame(data)
#df.to_csv('file.csv')
#df = pd.read_csv('file.csv')
print(df)
# CountVectorizer - класс конвертации текста в матрицу токенов
from sklearn.feature_extraction.text import CountVectorizer
# Создание матрицы признаков на основе мешка слов
count = CountVectorizer()
bag_of_words = count.fit_transform(df[1])
bag_of_words.toarray()
count.get_feature_names_out()  # Вывод имен признаков
# Создание матрицы признаков для 2-грамм
count_2gram = CountVectorizer(ngram_range=(2,2))
bag_of_2grams = count_2gram.fit_transform(df[1])
bag_of_2grams.toarray()
count_2gram.get_feature_names_out()  # Вывод имен признаков
# TfidfVectorizer - класс для преобразования текста в частотные векторы слов
from sklearn.feature_extraction.text import TfidfVectorizer
# Создание матрицы признаков
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(df[1])
feature_matrix.toarray()
# Матрица признаков
X = df[1]

# Вектор значений целевых переменных
y = df[0]


labels = df[0].unique() # Массив уникальных меток
# Разбиение на обучающую выборку (70%) и тестовую выборку (30%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train)
print(X_test)
print(y_test)
print(y_train)
# Загрузка Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Загрузка логистической регрессии
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])

logreg.fit(X_train, y_train) # Обучение модели
y_pred = logreg.predict(X_test)   # Предсказание на тестовых данных
# Оценка точности классификации
print('accuracy %s' % accuracy_score(y_pred, y_test))
yrt = open("test_dataset_test.csv", encoding='utf-8')
read_test = csv.reader(yrt, delimiter = ",")
ides=[]
tren_text=[]
for line in read_test:
    ID, text, topic, person = line
    if(m==1):
        m=100
    else:
        text=clean(text)
        topic=clean(topic)
        person=clean(person)
        tren_text.append(text+topic+person)
    ides.append(ID)
ides.pop(0)
for u in range(len(tren_text)):
    tokens = word_tokenize(tren_text[u])
    lemmed_tokens = [morph.parse(token)[0].normal_form for token in tokens if token not in russian_stopwords]
    tren_text[u] = " ".join(lemmed_tokens)
    lemm_texts_column.append(tren_text[u])
tren_text.pop(0)
categoriesss_answer = logreg.predict(tren_text)
df2 =pd.DataFrame(categoriesss_answer, index=ides, columns=[ 'Категория'])
df2.to_csv('Answer5.csv')


