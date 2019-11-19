from src.nets import ONet
from src.train import Trainer
import time
from tool import utils

def run():
    try:
        trainer = Trainer(net, netfile_name="onet_00_0", cfgfile='../src/cfg.ini')
        trainer.train()
    except(BaseException) as e:
        print("BaseException:", e, time.time())
        run()


if __name__ == '__main__':
    net = ONet()
    trainer = Trainer(net, netfile_name="onet_00_0", cfgfile='../src/cfg.ini')
    trainer.train()
    # run()
