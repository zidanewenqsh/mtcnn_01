from src.nets import PNet
from src.train import Trainer
import time
import sys
# from train import Trainer
import torch
from tool import utils
# from timekee
# p import Timekeeper


def run():
    try:
        trainer = Trainer(net, netfile_name="pnet_00_0", cfgfile='../src/cfg.ini')
        trainer.train()
    except(BaseException) as e:
        print("BaseException:", e, time.time())
        # with open("record.txt", 'a') as f:
        #     print("The onet already runned:", time.time(), file=f)
        run()


if __name__ == '__main__':
    net = PNet()
    # net.paraminit()
    # trainer = Trainer(net, save_path_o, label_path_48, pic_path_48, isCuda=True)
    # trainer.train()
    trainer = Trainer(net, netfile_name="pnet_00_0", cfgfile='../src/cfg.ini')
    trainer.train()
    # run()

