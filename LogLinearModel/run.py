from Config import *
from Log_Linear_Model import LogModel
from datetime import datetime

if __name__ == "__main__":
    start_time = datetime.now()
    print("#" * 10 + "开始训练" + "#" * 10)
    llm = LogModel(train_data_dir, dev_data_dir)
    llm.mini_batch_train(epoch, exitor, random_seed, shuffle_flag, C, eta)
    end_time = datetime.now()
    print("用时:" + str(end_time - start_time))