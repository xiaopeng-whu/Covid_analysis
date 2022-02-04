import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.sentiment.util import *

from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', None)

'''
    ----------
      数据清洗
    ----------
'''
path = "vaccine_side_effects.csv"
df = pd.read_csv(path, sep='\t')
# print(df.head(10))
# print(df.info())
df2 = df.dropna(axis=0, how='any', inplace=False)   # 将不包含attitude或side_effects的信息删除
# print(df2.head(10))
# print(df2.info())
df3 = df2.drop_duplicates(subset=['text'], keep=False)  # 将重复text的数据项删除便于训练
df4 = df3.reset_index(drop=True)    # 重置索引
# print(df4.head(10))
print(df4.info())

'''
    ------------------------------
      对text文本部分作进一步的预处理
    ------------------------------
'''
txt_df = df4['text']
# remove hashtags
txt_df2 = txt_df.apply(lambda x: re.sub(r"#\S+", "", str(x)))
# convert to lowercase
txt_df3 = txt_df2.apply(lambda x: x.lower())
# remove punctuations
txt_df4 = txt_df3.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
# remove special characters
txt_df5 = txt_df4.apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', str(x)))
nltk.download("stopwords")
# removing stopwords
stop_words = set(stopwords.words('english'))
txt_df6 = txt_df5.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
print(txt_df6.head(10))

'''
    ---------------------------------------------
        对数据集的attitude信息进行初步统计和可视化
    ---------------------------------------------
'''
# data = df4.attitude.value_counts()
# # print(data)
# plt.figure(figsize=(6, 6))
# plt.xlabel("Attitude", fontsize=12)
# plt.ylabel("Number", fontsize=12)
# plt.title("Sentiment Levels on "+str(path)+" dataset")
# sns.barplot(data.index, data.values)
# plt.show()

'''
    ---------------------------------------------
        对数据集的vaccine信息进行初步统计和可视化
    ---------------------------------------------
'''
# data2 = df4.vaccine.value_counts()
# # print(data2)
# plt.figure(figsize=(6, 6))
# plt.xlabel("Vaccine", fontsize=12)
# plt.ylabel("Number", fontsize=12)
# plt.title("Vaccine on "+str(path)+" dataset")
# sns.barplot(data2.index, data2.values)
# plt.show()

'''
    ------------------------------------------------------------------
        对数据集的side_effects信息进行初步统计、可视化和多标签数值化编码
    ------------------------------------------------------------------
'''
from sklearn.preprocessing import MultiLabelBinarizer
# 获取训练集合、测试集的事件类型
labels = []
side_effects_list = []
for line in df4['side_effects']:
    genres = line.split(",")
    labels.append(genres)
    side_effects_list.extend(genres)
# 利用sklearn中的MultiLabelBinarizer进行多标签编码
mlb = MultiLabelBinarizer()
mlb.fit(labels)
# print("一共有%d种事件类型。" % len(mlb.classes_))

# 可视化
x = ['Headache', 'Redness', 'Swelling', 'Tiredness', 'Muscle Pain', 'Chills', 'Fever', 'Nausea', 'None', 'No Details', 'Other']
y = [side_effects_list.count(i) for i in x]
plt.figure(figsize=(6, 6))
plt.xlabel("Side effects", fontsize=12)
plt.ylabel("Number", fontsize=12)
plt.title("The number of side_effects on "+str(path)+" dataset")
sns.barplot(x, y)
plt.show()

# 进行多标签编码
labels1 = []
for line in df4['side_effects']:
    genres = line.split(",")
    labels1.append(mlb.transform([genres])[0])
labels1 = np.array(labels1)

'''
    ------------------------------
        对label进行数值化处理
    ------------------------------
'''
df4['attitude'].replace(['Negative','Neutral','Positive'],[0,1,2],inplace=True)
# print(df4.head(10))
df4['vaccine'].replace(['Pfizer','Moderna','AstraZeneca','Sinovac'],[0,1,2,3],inplace=True)

'''
    --------------------------------------------------------------------------------
    以4：1的比例划分训练集和测试集，
        使用不同的特征提取方式：BOW、TF-IDF、keras.preprocessing.text.Tokenizer，
        使用多种分类器：多项式朴素贝叶斯、SVM、决策树、神经网络（CNN、RNN、Bi-GRU等组合）
    进行训练测试
    --------------------------------------------------------------------------------
'''
# 1 bag-of-words feature matrix
print("----------CountVectorizer----------")
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(stop_words='english')
X = bow_vectorizer.fit_transform(txt_df6)
y = df4["attitude"]
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(X_train[0], y_train[0])

# 1.1 多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("------MultinomialNB------")
print("Training accuracy: ", classifier.score(X_train, y_train))
print("Test accuracy: ", classifier.score(X_test, y_test))

# 1.2 SVM
from sklearn.svm import LinearSVC
LSVC = LinearSVC()
LSVC.fit(X_train, y_train)
y_pred2 = LSVC.predict(X_test)
print("------LinearSVC------")
print("Training accuracy: ", LSVC.score(X_train, y_train))
print("Test accuracy: ", LSVC.score(X_test, y_test))

# 1.3 Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)
y_pred3 = DTC.predict(X_test)
print("------DecisionTree------")
print("Training accuracy: ", DTC.score(X_train, y_train))
print("Test accuracy: ", DTC.score(X_test, y_test))


# 2 TF-IDF feature matrix
from sklearn.feature_extraction.text import TfidfVectorizer
print("----------TfidfVectorizer----------")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(txt_df6)
y = df4["attitude"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)

# 2.1 多项式朴素贝叶斯
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred4 = classifier.predict(X_test)
print("------MultinomialNB------")
print("Training accuracy: ", classifier.score(X_train, y_train))
print("Test accuracy: ", classifier.score(X_test, y_test))

# 2.2 SVM
LSVC = LinearSVC()
LSVC.fit(X_train, y_train)
y_pred5 = LSVC.predict(X_test)
print("------LinearSVC------")
print("Training accuracy: ", LSVC.score(X_train, y_train))
print("Test accuracy: ", LSVC.score(X_test, y_test))

# 2.3 Decision Tree
DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)
y_pred6 = DTC.predict(X_test)
print("------DecisionTree------")
print("Training accuracy: ", DTC.score(X_train, y_train))
print("Test accuracy: ", DTC.score(X_test, y_test))

# 3 神经网络
import keras
from keras.models import Sequential,Model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, Input, Lambda, Reshape, Activation, BatchNormalization
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional

from gensim.models import Word2Vec  # gensim4的用法

feature = txt_df6
y = df4["attitude"]
y2 = df4["vaccine"]
# 训练模型，词向量的长度设置为500，采用skip-gram模型，采用负采样，窗口选择6，最小词频是7，模型保存为pkl格式
w2v_model=Word2Vec(sentences=feature, vector_size=500, sg=1,hs=0,window=6, min_count=7)
w2v_model.wv.save_word2vec_format("./word2Vec" + ".pkl", binary=True)

NUM_CLASS = 3       # 态度数量
NUM_CLASS2 = 4      # 疫苗种类数量
NUM_CLASS3 = len(mlb.classes_)  # 副作用种类数量
INPUT_SIZE = 64     # 输入维度
# # 序列对齐文本数据
# Tokenizer是一个用于向量化文本，或将文本转换为序列
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tokenizer.fit_on_texts(feature)
vocab = tokenizer.word_index
# print("vocab:", len(vocab)) # 5725
x_ids = tokenizer.texts_to_sequences(feature)
pad_s = pad_sequences(x_ids, maxlen=INPUT_SIZE)
target_u = to_categorical(y, NUM_CLASS)
X_train, X_test, y_train, y_test = train_test_split(pad_s, target_u, random_state=22, test_size=0.2)
# target_u2 = to_categorical(y2, NUM_CLASS2)
# X_train2, X_test2, y_train2, y_test2 = train_test_split(pad_s, target_u2, random_state=22, test_size=0.2)
# target_u3 = labels1
# X_train3, X_test3, y_train3, y_test3 = train_test_split(pad_s, target_u3, random_state=22, test_size=0.2)

embedding_matrix = np.zeros((len(vocab)+1, 500))
for word, i in vocab.items():
    try:
        embedding_vector=w2v_model.wv[str(word)]
        embedding_matrix[i]=embedding_vector
    except:
        print("Word: [",word,"] not in wvmodel! Use random embedding instead.")

main_input = Input(shape=(INPUT_SIZE,), dtype='float64')
model = Sequential()

# 3.1 word2vec+RNN
model.add(Embedding(len(vocab)+1, 500, input_length=INPUT_SIZE, weights=[embedding_matrix], trainable=True))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.1))
model.add(Dense(NUM_CLASS, activation='softmax'))
# model.add(Dense(NUM_CLASS2, activation='softmax'))
# model.add(Dense(NUM_CLASS3, activation='sigmoid'))  # 副作用模型使用sigmoid

# # 3.2 word2vec+CNN+GRU
# model.add(Embedding(len(vocab)+1, 500, input_length=INPUT_SIZE, weights=[embedding_matrix], trainable=True))
# model.add(Convolution1D(256, 3, padding='same', strides=1))
# model.add(Activation('relu'))
# model.add(MaxPool1D(pool_size=2))
# model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
# model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1))
# model.add(Dense(NUM_CLASS, activation='softmax'))
# # model.add(Dense(NUM_CLASS2, activation='softmax'))
# # model.add(Dense(NUM_CLASS3, activation='sigmoid'))
#
# # 3.3 word2vec+Bi-GRU
# model.add(Embedding(len(vocab)+1, 500, input_length=INPUT_SIZE, weights=[embedding_matrix], trainable=True))
# model.add(Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True)))
# model.add(Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.1)))
# model.add(Dense(NUM_CLASS, activation='softmax'))
# # model.add(Dense(NUM_CLASS2, activation='softmax'))
# # model.add(Dense(NUM_CLASS3, activation='sigmoid'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   # 副作用模型
print('Train...')
model.fit(X_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=[X_test, y_test])
loss, accuracy = model.evaluate(X_test, y_test, batch_size=16)
model.save('model.h5')
# model.fit(X_train2, y_train2,
#           batch_size=32,
#           epochs=10,
#           validation_data=[X_test2, y_test2])
# model.save('model2.h5')
# model.fit(X_train3, y_train3,
#           batch_size=32,
#           epochs=10,
#           validation_data=[X_test3, y_test3])
# model.save('model3.h5')
