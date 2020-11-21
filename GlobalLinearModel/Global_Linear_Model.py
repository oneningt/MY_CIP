import numpy as np
import random
from Data_Reader import DataReader
from datetime import datetime


class GlobalLinearModel:
    def __init__(self, train_data, dev_data):
        self.train_data = DataReader(train_data)
        self.dev_data = DataReader(dev_data)
        self.matrix_weight = 0
        self.v_matrix_weight = 0
        self.update_times = 0
        self.update_time = 0
        self.feature_set = set()
        self.feature_id = {}
        self.bi_gram_features = []
        self.create_feature_space()

    @staticmethod
    def create_feature_template(sentence_word, pos, prev_tag=None, gram='U'):
        """
        创建部分特征模板
        :param gram:'a'为带前一词词性，'U'为仅当前词词性
        :param prev_tag: 前一词词性
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

        if gram != 'U':
            feature_list.append(('01', prev_tag))
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
        for sentence_word, sentence_tag in self.train_data.sentences_total:
            for i in range(len(sentence_word)):
                prev_tag = sentence_tag[i - 1] if i > 0 else '  '
                feature_list = self.create_feature_template(sentence_word, i, prev_tag, gram='a')
                for feature in feature_list:
                    self.feature_set.add(feature)

        self.feature_id = {feature: feature_id for feature_id, feature in enumerate(list(sorted(self.feature_set)))}
        self.bi_gram_features = [[('01', tag)] for tag in self.train_data.tags]  # 列向量
        self.matrix_weight = np.zeros((len(self.feature_set), self.train_data.tag_num))
        self.v_matrix_weight = np.zeros((len(self.feature_set), self.train_data.tag_num))
        self.update_times = np.zeros((len(self.feature_set), self.train_data.tag_num), dtype='int')

    def score(self, feature_list):
        """
        将该词标为所有词性的得分
        :param feature_list:
        :return:
        """
        feature_list_id = [self.feature_id[feature] for feature in feature_list if feature in self.feature_set]
        score_matrix = self.v_matrix_weight[feature_list_id]
        score_list = np.sum(score_matrix, axis=0)

        return score_list

    def predict(self, sentence_word: [int]):
        """
        预测正确的词性
        :param sentence_word: 分词的句子
        :return: 预测的词性句子
        """
        word_num = len(sentence_word)
        dp_matrix = np.zeros((word_num, self.train_data.tag_num))
        backward_trace_matrix = np.zeros((word_num, self.train_data.tag_num), dtype=int)
        bi_scores = np.array([self.score(bi_feature) for bi_feature in self.bi_gram_features])

        dp_matrix[0] = self.score(self.create_feature_template(sentence_word, 0, '<BOS>', gram='a'))
        backward_trace_matrix[0] = np.full([self.train_data.tag_num], -1)

        for i in range(1, word_num):
            prev_tag = self.train_data.tags[np.argmax(dp_matrix[i - 1])]
            line_score = self.score(self.create_feature_template(sentence_word, i, gram='U'))
            scores = np.array((dp_matrix[i - 1] + bi_scores.T).T + line_score)
            backward_trace_matrix[i] = np.argmax(scores, axis=0)
            dp_matrix[i] = np.max(scores, axis=0)

        prev = np.argmax(dp_matrix[-1])
        result = [prev]

        for i in range(word_num - 1, 0, -1):
            prev = backward_trace_matrix[i][prev]
            result.append(prev)

        return list(map(lambda x: self.train_data.tags[x], result[::-1]))

    def evaluate(self, dataset: DataReader):
        """
        评价函数
        :param dataset:
        :return: 返回
        """
        correct_num = 0
        for pos, sentence_word in enumerate(dataset.sentences_word):
            correct_tag = dataset.sentences_tag[pos]
            predict_tag = self.predict(sentence_word)
            correct_num += len([i for i in range(len(correct_tag)) if correct_tag[i] == predict_tag[i]])

        accuracy = correct_num / dataset.word_num
        return dataset.word_num, correct_num, accuracy

    def update(self, sentence):
        """
        更新权重
        :param sentence:
        :return:
        """
        sentence_word = sentence[0]
        correct_sentence_tag = sentence[-1]
        predict_tag_list = self.predict(sentence_word)
        if predict_tag_list != correct_sentence_tag:
            for i in range(len(sentence_word)):
                prev_tag = predict_tag_list[i - 1] if i > 0 else '<BOS>'
                if predict_tag_list[i] != correct_sentence_tag[i]:
                    predict_tag_id = self.train_data.tag_id[predict_tag_list[i]]
                    correct_tag_id = self.train_data.tag_id[correct_sentence_tag[i]]
                    word_feature = self.create_feature_template(sentence_word, i, prev_tag, gram='a')
                    word_feature_id = [self.feature_id[feature] for feature in word_feature if
                                       feature in self.feature_set]
                    for feature_id in word_feature_id:
                        last_weight = self.matrix_weight[feature_id][predict_tag_id]
                        self.matrix_weight[feature_id][predict_tag_id] -= 1
                        self.v_matrix_weight[feature_id][predict_tag_id] += (self.update_time - self.update_times[feature_id][predict_tag_id] - 1) * last_weight + self.matrix_weight[feature_id][
                            predict_tag_id]
                        self.update_times[feature_id][predict_tag_id] = self.update_time

                        last_weight = self.matrix_weight[feature_id][correct_tag_id]
                        self.matrix_weight[feature_id][correct_tag_id] += 1
                        self.v_matrix_weight[feature_id][correct_tag_id] += (self.update_time - self.update_times[feature_id][correct_tag_id] - 1) * last_weight + \
                                                                            self.matrix_weight[feature_id][
                                                                                correct_tag_id]
                        self.update_times[feature_id][correct_tag_id] = self.update_time

    def train(self, epoch=100, exitor=10, random_seed=0, shuffle_flag=True):
        """
        :param epoch: 最大迭代次数
        :param exitor: 退出轮数
        :param random_seed: 随机种子
        :param shuffle_flag: 是否打乱数据
        :return:
        """
        dev_max_epoch = 0
        dev_max_accuracy = 0

        random.seed(random_seed)

        if shuffle_flag:
            random.shuffle(self.train_data.sentences_total)

        for iteration_num in range(1, epoch + 1):
            print("第{:d}轮训练".format(iteration_num))
            start_time = datetime.now()
            for sentence in self.train_data.sentences_total:
                self.update(sentence)
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
