from src.nets import RNet
# from test01.nets import RNet
from src.train import Trainer
import time
from tool import utils

# import torch


def run():
    try:
        trainer = Trainer(net, netfile_name="rnet_00_0", cfgfile='../src/cfg.ini')
        trainer.train()
    except(BaseException) as e:
        print("BaseException:", e, time.time())
        # with open("record.txt", 'a') as f:
        #     print("The onet already runned:", time.time(), file=f)
        run()


if __name__ == '__main__':
    net = RNet()
    # net.paraminit()
    # trainer = Trainer(net, save_path_o, label_path_48, pic_path_48, isCuda=True)
    # trainer.train()
    trainer = Trainer(net, netfile_name="rnet_00_0", cfgfile='../src/cfg.ini')
    trainer.train()
    # run()

