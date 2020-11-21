from Config import *
from Global_Linear_Model import GlobalLinearModel
from datetime import datetime

if __name__ == "__main__":
    start_time = datetime.now()
    print("#" * 10 + "开始训练" + "#" * 10)
    llm = GlobalLinearModel(train_data_dir, dev_data_dir)
    llm.train(epoch, exitor, random_seed, shuffle_flag)
    end_time = datetime.now()
    print("用时:" + str(end_time - start_time))