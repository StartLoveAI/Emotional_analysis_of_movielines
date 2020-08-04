# -*- coding: utf-8 -*-
"""
数据集说明：数据来源于电影中的台词文本。文件positive.txt, negative.txt分别存储
有5331条正面情感的台词文本数据，331条负面情感的台词文本数据。
程序说明：训练模型，对输入的文本进行情绪分析。在现实中有现成的NLP库可以处理，譬如TextBlob
"""

import os
import random
import pickle
import codecs
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf


def create_lexicon(pos, neg, hm_lines=10000000, word_frequency_min=50, word_frequency_max=1000):
    """
    通过解析pos和neg文本，生成一个字典
    :param pos: 积极文本。
    :param neg: 消极文本。
    :param hm_lines: 支持的最大行数。
    :param word_frequency_min: 可放入字典中的单词在文本中出现的最小次数。
    :param word_frequency_max: 可放入字典中的单词在文本中出现的最大次数。
    :return: 返回生成的字典
    """
    lexicon = []

    for text_file in [pos, neg]:
        with codecs.open(text_file, 'r', "latin-1") as f:  # 输入文件是以latin编码的，需要用响应的解码器
            lines = f.readlines()
            for line in lines[:hm_lines]:
                all_words = word_tokenize(line.lower())  # word_tokenize是分词器
                lexicon += list(all_words)

    # 词规范化(还原词本身)：如把broken替换为break，buses替换成bus
    lexicon = [WordNetLemmatizer().lemmatize(index) for index in lexicon]

    # Counter类的目的是用来跟踪值出现的次数。它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key，其计数作为value
    w_counts = Counter(lexicon)

    l2 = []
    for w in w_counts:
        if w_counts[w] in range(word_frequency_min, word_frequency_max):
            l2.append(w)

    print("The lexicon size is: %s" % len(l2))

    return l2


def sample_handling(sample, hm_lines, lexicon, classification):
    """
    将sample文本，逐行转换成特征向量。
    :param sample: 输入文本
    :param hm_lines: 支持的最大行数。
    :param lexicon: 字典
    :param classification: 文本分类列表，如[1, 0]代表积极
    :return: 返回特征向量集
    """
    features_set = []

    with codecs.open(sample, 'r', "latin-1") as sam:
        lines = sam.readlines()
        for line in lines[:hm_lines]:
            current_words = [WordNetLemmatizer().lemmatize(index)
                             for index in word_tokenize(line.lower())]
            features = np.zeros(len(lexicon))
            for w in current_words:
                if w.lower() in lexicon:
                    index = lexicon.index(w.lower())
                    features[index] += 1

            features = list(features)

            features_set.append([features, classification])

    return features_set


def create_features_and_labels_set(pos, neg, hm_lines=10000000, word_frequency_min=50,
                                   word_frequency_max=1000, test_size=0.2,
                                   dataset_name='sentiment_dataset.pickle'):
    """
    将文本处理成机器学习算法可以使用的特征向量数据集。
    :param pos: 积极文本。
    :param neg: 消极文本。
    :param hm_lines: 支持的最大行数。
    :param word_frequency_min: 可放入字典中的单词在文本中出现的最小次数。
    :param word_frequency_max: 可放入字典中的单词在文本中出现的最大次数。
    :param test_size: 测试集的比例大小设置
    :param dataset_name: 获取数据集，将其保存的名称
    :return: x_train, y_train, x_test, y_test
    """
    if os.path.exists(dataset_name):
        print("loading dataset...")
        try:
            with open(dataset_name, 'rb') as dataset:
                return pickle.load(dataset)
        except Exception:
            print("try to created again, due to: '%s'" % e)

    features = []

    lexicon = create_lexicon(pos, neg, hm_lines, word_frequency_min, word_frequency_max)
    features += sample_handling(pos, hm_lines, lexicon, [1, 0])
    features += sample_handling(neg, hm_lines, lexicon, [0, 1])

    random.shuffle(features)

    features = np.array(features)

    test_size = int(test_size * len(features))

    x_train = list(features[:, 0][:-test_size])
    y_train = list(features[:, 1][:-test_size])
    x_test = list(features[:, 0][-test_size:])
    y_test = list(features[:, 1][-test_size:])

    with open(dataset_name, 'wb') as dataset:
        pickle.dump([x_train, y_train, x_test, y_test], dataset)

    return x_train, y_train, x_test, y_test


def weight_variable(shape):
    """权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。"""
    initial = tf.truncated_normal(shape=shape, stddev=0.1)

    return tf.Variable(initial)


def bias_variable(shape):
    """ReLu使用一个较小的正数来初始化偏置，以避免神经元输出恒为0"""
    initial = tf.constant(shape=shape, value=0.1)

    return tf.Variable(initial)


# sizes=[433, 100, 100, 100, 2]
def neural_network_model(data, sizes=list()):
    num_layer = len(sizes)
    layer_parameters = []

    for index in range(num_layer - 1):
        parameters = {'weight': weight_variable([sizes[index], sizes[index + 1]]),
                      'bias': bias_variable([sizes[index + 1]])}
        layer_parameters.append(parameters)

    for index in range(len(layer_parameters)):
        data = tf.add(tf.matmul(data, layer_parameters[index]['weight']),
                      layer_parameters[index]['bias'])
        data = tf.nn.relu(data)

    return data


def train_neural_network(sizes=(433, 2), batch_size=128, hm_epochs=10):
    x = tf.placeholder(tf.float32, [None, len(train_x[0])])
    y = tf.placeholder(tf.float32, [None, len(train_y[0])])

    prediction = neural_network_model(x, list(sizes))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, np.float32))

        print('Accuracy:', accuracy.eval({x: np.array(test_x), y: np.array(test_y)}))


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_features_and_labels_set('positive.txt', 'negative.txt')
    train_neural_network(sizes=(433, 10, 10, 2), batch_size=128, hm_epochs=10)
