import numpy as np
import random
from Config import *
from Data_Reader import DataReader
from collections import defaultdict
from scipy.special import logsumexp
from datetime import datetime


class LogModel:
    def __init__(self, train_data, dev_data):
        self.train_data = DataReader(train_data)
        self.dev_data = DataReader(dev_data)
        self.matrix_weight = 0
        self.feature_set = set()
        self.feature_id = {}
        self.create_feature_space()

    @staticmethod
    def create_feature_template(sentence_word, pos):
        """
        创建部分特征模板
        :param sentence_word: 句子分词列表
        :param pos: 需要创建模板的词的索引
        :return:
        ft_list：部分特征模板
        """
        feature_list = []
        word = sentence_word[pos]
        prev_word = sentence_word[pos - 1] if pos > 0 else "**"
        next_word = sentence_word[pos + 1] if pos + 1 < len(sentence_word) else "##"
        word_first_char = word[0]
        word_last_char = word[-1]
        prev_word_last_char = prev_word[-1]
        next_word_first_char = next_word[0]

        feature_list.append(('02', word))
        feature_list.append(('03', prev_word))
        feature_list.append(('04', next_word))
        feature_list.append(('05', word, prev_word_last_char))
        feature_list.append(('06', word, next_word_first_char))
        feature_list.append(('07', word_first_char))
        feature_list.append(('08', word_last_char))

        for char in word[1:-1]:
            feature_list.append(('09', char))
            feature_list.append(('10', word_first_char, char))
            feature_list.append(('11', word_last_char, char))

        if 1 == len(word):
            feature_list.append(('12', word, prev_word_last_char, next_word_first_char))

        for i in range(1, len(word)):
            if word[i - 1] == word[i]:
                feature_list.append(('13', word[i - 1], 'consecutive'))
            if i <= 4:
                feature_list.append(('14', word[:i]))
                feature_list.append(('15', word[-i:]))

        if len(word) <= 4:
            feature_list.append(('14', word))
            feature_list.append(('15', word))

        return feature_list

    def create_feature_space(self):
        """
        创建特征空间，即self.matrix_weight参数
        :return:
        """
        features_set = set()
        for sentence_word in self.train_data.sentences_word:
            for i in range(len(sentence_word)):
                for feature in self.create_feature_template(sentence_word, i):
                    self.feature_set.add(feature)

        self.feature_id = {feature: feature_id for feature_id, feature in enumerate(list(self.feature_set))}
        self.matrix_weight = np.zeros((len(self.feature_set), self.train_data.tag_num))

    def predict(self, sentence_word, pos):
        """
        预测正确的词性
        :param sentence_word: 分词的句子
        :param pos: 要预测词性的词
        :return: 预测的词性
        """
        feature_list = self.create_feature_template(sentence_word, pos)
        feature_list_id = [self.feature_id[feature] for feature in feature_list if feature in self.feature_set]
        score_matrix = self.matrix_weight[feature_list_id]
        score_list = np.sum(score_matrix, axis=0)

        return self.train_data.tags[np.argmax(score_list)]

    def evaluate(self, dataset: DataReader):
        """
        评价函数
        :param dataset:
        :return: 返回
        """
        correct_num = 0
        for sentence_pos, sentence_word in enumerate(dataset.sentences_word):
            sentence_tag = dataset.sentences_tag[sentence_pos]
            for pos in range(len(sentence_word)):
                if self.predict(sentence_word, pos) == sentence_tag[pos]:
                    correct_num += 1
        accuracy = correct_num / dataset.word_num
        return dataset.word_num, correct_num, accuracy

    def mini_batch_train(self, epoch=100, exitor=10, random_seed=0, shuffle_flag=True, lmbda=0.01, eta=0.5):
        """
        按照文档中选取一定数据进行梯度下降，更像mini——batch,而不是随机下降
        :param epoch: 最大迭代次数
        :param exitor: 退出轮数
        :param random_seed: 随机种子
        :param shuffle_flag: 是否打乱数据
        :return:
        """
        dev_max_epoch = 0
        dev_max_accuracy = 0
        decay_rate = 0.96
        step = 1
        learning_rate = eta

        random.seed(random_seed)

        train_data = []
        for sentence_id in range(len(self.train_data.sentences)):
            sentence_word = self.train_data.sentences_word[sentence_id]
            sentence_tag = self.train_data.sentences_tag[sentence_id]
            for pos, word in enumerate(sentence_word):
                feature_list = self.create_feature_template(sentence_word, pos)
                train_data.append((word, feature_list, sentence_tag[pos]))

        if shuffle_flag:
            random.shuffle(train_data)

        batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]

        for iteration_num in range(1, epoch + 1):
            print("第{:d}轮训练".format(iteration_num))
            start_time = datetime.now()
            for batch in batches:
                self.gradient_descent(batch, eta * decay_rate ** (step / 10000), lmbda)
                step += 1
            end_time = datetime.now()
            print("用时:" + str(end_time - start_time))
            print("训练完毕")
            train_total_num, train_correct_num, train_accuracy = self.evaluate(self.train_data)
            print("训练集共{:d}个词，正确判断{:d}个词，正确率{:f}".format(train_total_num, train_correct_num, train_accuracy))
            dev_total_num, dev_correct_num, dev_accuracy = self.evaluate(self.dev_data)
            print("开发集共{:d}个词，正确判断{:d}个词，正确率{:f}".format(dev_total_num, dev_correct_num, dev_accuracy))

            if dev_accuracy > dev_max_accuracy:
                dev_max_epoch = iteration_num
                dev_max_accuracy = dev_accuracy
            elif iteration_num - dev_max_epoch >= exitor:
                print("经过{:d}轮训练正确率无提升，结束训练，最大正确率为第{:d}轮训练后的{:f}".format(exitor, dev_max_epoch, dev_max_accuracy))
                break
            print()

    def gradient_descent(self, batch, learning_rate, lmbda=0.01):
        gradients = defaultdict(float)

        for word, feature_list, sentence_tag in batch:
            feature_list_id = [self.feature_id[feature] for feature in feature_list if feature in self.feature_set]
            score_matrix = self.matrix_weight[feature_list_id]
            score_list = np.sum(score_matrix, axis=0)
            possibility = np.exp(score_list - logsumexp(score_list))

            for feature_id in feature_list_id:
                gradients[feature_id] -= possibility
                gradients[feature_id, self.train_data.tag_id[sentence_tag]] += 1
        for key, gradient in gradients.items():
            self.matrix_weight[key] -= learning_rate * (lmbda * self.matrix_weight[key] - gradient)
