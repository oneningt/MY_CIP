class DataReader:
    def __init__(self, data_file):
        self.sentences = []  # 以（词，词性）的方式放置
        self.sentences_word = []  # 仅分词
        self.sentences_tag = []  # 仅词性
        self.tags = []  # 所有出现词性集合
        self.tag_id = {}  # 词性 序号
        self.word_num = 0
        self.tag_num = 0
        self.data_reader(data_file)

    def data_reader(self, data_file):
        sentence = []
        sentence_word = []
        sentence_tag = []
        tag_set = set()
        with open(data_file, encoding="UTF-8") as file:
            for line in file:
                if line == "\n":
                    self.sentences.append(sentence)
                    self.sentences_word.append(sentence_word)
                    self.sentences_tag.append(sentence_tag)
                    sentence = []
                    sentence_word = []
                    sentence_tag = []
                else:
                    split_line = line.split()
                    word = split_line[1]
                    tag = split_line[3]
                    sentence.append((word, tag))
                    sentence_word.append(word)
                    sentence_tag.append(tag)
                    tag_set.add(tag)
                    self.word_num += 1

        self.tags = list(sorted(tag_set))
        self.tag_id = {tag: tag_id for tag_id, tag in enumerate(self.tags)}
        self.tag_num = len(self.tags)
