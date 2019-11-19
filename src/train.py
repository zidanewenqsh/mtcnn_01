import sys
import os


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.nets import PNet, RNet, ONet, Net
from tool import *

from detect.mtcnndetect import Detector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFGFILE = ".\src\cfg.ini"

SAVE_DIR = r"..\save\20190910"
PIC_DIR = r"D:\datasets\save_10261_20190725\pic"
LABEL_DIR = r"D:\datasets\save_10261_20190725\label"
NETFILE_EXTENTION = "pt"

ALPHA = 0.5

CONTINUETRAIN = True
NEEDTEST = False
NEEDSAVE = False
NEEDSHOW = False
EPOCH = 10000
BATCHSIZE = 256
NUMWORKERS = 4
LR = 1e-3
ISCUDA = True
SAVEDIR_EPOCH = r"..\save\netbackup"
TEST_IMG = r"../test/005290.jpg"
PRETRAINED_PNET = r"../param/pnet_07.pth"
PRETRAINED_RNET = r"../param/rnet_07_4.pth"
PRETRAINED_ONET = r"../param/onet_07_4.pth"
RECORDPOINT = 10
TESTPOINT = 100

class Trainer:
    def __init__(self, net: Net, netfile_name, cfgfile=None):
        self.net = net
        self.netfile_name = netfile_name
        if cfgfile != None:
            self.cfginit(cfgfile)
        utils.makedir(SAVE_DIR)
        parser = argparse.ArgumentParser(description="base class for network training")
        self.args = self.argparser(parser)
        net_savefile = "{0}.{1}".format(self.netfile_name, NETFILE_EXTENTION)
        self.save_dir = os.path.join(SAVE_DIR, "nets")
        utils.makedir(self.save_dir)
        self.save_path = os.path.join(self.save_dir, net_savefile)
        self.savepath_epoch = os.path.join(SAVEDIR_EPOCH, net_savefile)
        self.size = {"Pnet": 12, "Rnet": 24, "Onet": 48}[self.net.name]
        self.cls_loss = nn.BCELoss()
        self.offset_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())

        if ISCUDA:
            self.net = self.net.to(DEVICE)

        if NEEDTEST:
            self.detecter = Detector(
                returnnet=self.net.name,
                trainnet=self.net
            )
        self.logdir = os.path.join(SAVE_DIR, "log")

        utils.makedir(self.logdir)
        self.logfile = os.path.join(self.logdir, "{0}.txt".format(self.netfile_name))
        if not os.path.exists(self.logfile):
            with open(self.logfile, 'w') as f:
                print("%.2f %d    " % (0.00, 0), end='\r', file=f)
                print("logfile created")

        print("initial complete")

    def cfginit(self, cfgfile):
        config = configparser.ConfigParser()
        config.read(cfgfile)
        items_ = config.items(self.netfile_name)
        for key, value in items_:
            if key.upper() in globals().keys():
                try:
                    globals()[key.upper()] = config.getint(self.netfile_name, key.upper())
                except:
                    try:
                        globals()[key.upper()] = config.getfloat(self.netfile_name, key.upper())
                    except:
                        try:
                            globals()[key.upper()] = config.getboolean(self.netfile_name, key.upper())
                        except:
                            globals()[key.upper()] = config.get(self.netfile_name, key.upper())

    def argparser(self, parser):
        """default argparse, please customize it by yourself. """

        parser.add_argument("-e", "--epoch", type=int, default=EPOCH, help="number of epochs")
        parser.add_argument("-b", "--batch_size", type=int, default=BATCHSIZE, help="mini-batch size")
        parser.add_argument("-n", "--num_workers", type=int, default=NUMWORKERS,
                            help="number of threads used during batch generation")
        parser.add_argument("-l", "--lr", type=float, default=LR, help="learning rate for gradient descent")
        parser.add_argument("-r", "--record_point", type=int, default=RECORDPOINT, help="print frequency")
        parser.add_argument("-t", "--test_point", type=int, default=TESTPOINT,
                            help="interval between evaluations on validation set")
        parser.add_argument("-a", "--alpha", type=float, default=ALPHA, help="ratio of conf and offset loss")
        return parser.parse_args()

    def loss_fn(self, output_cls, output_offset, cls, offset):

        cls_loss = self.cls_loss(output_cls, cls)
        offset_loss = self.offset_loss(offset, output_offset)
        loss = ALPHA * cls_loss + (1 - ALPHA) * offset_loss
        return loss, cls_loss, offset_loss

    def logging(self, result, dataloader_len, RECORDPOINT):

        with open(self.logfile, "r+") as f:

            if f.readline() == "":
                batchcount = 0
                f.seek(0, 0)
                print("%.2f %d    " % (0.00, 0), end='\r', file=f)

            else:
                f.seek(0, 0)

                batchcount = int(f.readline().split()[-1]) + RECORDPOINT

            f.seek(0, 0)
            print("%.2f %d " % (batchcount / dataloader_len, batchcount), end='', file=f)

            f.seek(0, 2)
            print(result, file=f)

    def getstatistics(self):
        datalist = []
        with open(self.logfile) as f:
            for line in f.readlines():
                if not line[0].isdigit():
                    datalist.append(eval(line))
        return datalist

    def scalarplotting(self, datalist, key):
        save_dir = os.path.join(SAVE_DIR, key)
        utils.makedir(save_dir)
        save_name = "{0}.jpg".format(key)

        save_file = os.path.join(save_dir, save_name)
        values = []
        for data_dict in datalist:
            if data_dict:
                values.append(data_dict[key])
        if len(values) != 0:
            plt.plot(values)
            plt.savefig(save_file)
            plt.show()

    def FDplotting(self, net: Net):
        save_dir = os.path.join(SAVE_DIR, "params")
        utils.makedir(save_dir)
        save_name = "{0}_param.jpg".format(self.netfile_name)
        save_file = os.path.join(SAVE_DIR, save_name)
        params = []
        for param in net.parameters():
            params.extend(param.view(-1).cpu().detach().numpy())
        params = np.array(params)
        histo = np.histogram(params, 10, range=(np.min(params), np.max(params)))
        plt.plot(histo[1][1:], histo[0])
        plt.savefig(save_file)
        plt.show()

    def train(self):
        start_time = time.time()
        facedataset = faceDatasets.FaceDataset(LABEL_DIR, PIC_DIR, self.size)
        dataloader = data.DataLoader(facedataset, batch_size=self.args.batch_size, shuffle=True,
                                     num_workers=self.args.num_workers,
                                     drop_last=True)
        dataloader_len = len(dataloader)

        if os.path.exists(self.logfile):
            with open(self.logfile) as f:
                if f.readline() != "":
                    f.seek(0, 0)
                    batch_count = int(float(f.readline().split()[1]))

        for i in range(self.args.epoch):
            self.net.train()
            for j, (img_data_, cls_, offset_) in enumerate(dataloader):
                self.net.train()
                if ISCUDA:
                    img_data_ = img_data_.to(DEVICE)
                    cls_ = cls_.to(DEVICE)
                    offset_ = offset_.to(DEVICE)
                _output_cls, _output_offset = self.net(img_data_)
                _output_cls = _output_cls.view(-1, 1)
                _output_offset = _output_offset.view(-1, 4)

                cls_mask = torch.lt(cls_[:, 0], 2)
                offset_mask = torch.gt(cls_[:, 0], 0)

                cls = cls_[cls_mask]
                offset = offset_[offset_mask]

                output_cls = _output_cls[cls_mask]
                output_offset = _output_offset[offset_mask]

                loss, cls_loss, offset_loss = self.loss_fn(output_cls, output_offset, cls, offset)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                params = []
                for param in self.net.parameters():
                    params.extend(param.view(-1).data)

                checktime = time.time() - start_time
                print("j", j, self.args.record_point)
                if j % self.args.record_point == 0:

                    cls_acc = torch.mean(torch.lt(torch.abs(torch.sub(cls, output_cls)), 0.02).float())
                    offset_acc = torch.mean(torch.lt(torch.abs(torch.sub(offset, output_offset)), 0.02).float())

                    result = "{'epoch':%d,'batch':%d,'loss':%f,'cls_loss':%f,'offset_loss':%f,'total_time':%.2f,'cls_acc':%f,'offset_acc':%f,'time':%s}" % (
                        i, j, loss, cls_loss, offset_loss, checktime, cls_acc, offset_acc,
                        time.strftime("%Y%m%d%H%M%S", time.localtime()))
                    print(result)
                    self.logging(result, dataloader_len, self.args.record_point)
                    if NEEDSAVE:
                        torch.save(self.net.state_dict(), self.save_path)
                        print("net save successful")

                if NEEDTEST and j % self.args.test_point == 0:
                    self.test(batch_count)

            if NEEDSAVE:
                torch.save(self.net.state_dict(), self.savepath_epoch)
                print("an epoch save successful")

    def test(self, batch_count):
        with torch.no_grad():
            self.net.eval()
            print(TEST_IMG)
            img = Image.open(TEST_IMG)
            boxes = self.detecter.detect(img)
            draw = ImageDraw.Draw(img)
            for box in boxes:
                draw.rectangle(box, outline="red", width=5)

            if NEEDSAVE:
                testpic_savedir = os.path.join(SAVE_DIR, "testpic", self.netfile_name)
                utils.makedir(testpic_savedir)
                print(testpic_savedir)
                testpic_savefile = os.path.join(testpic_savedir, "{0}.jpg".format(batch_count))
                img.save(testpic_savefile)
                print("testpic save successful")

            if NEEDSHOW:
                plt.clf()
                plt.axis("off")
                plt.imshow(img)
                plt.pause(0.1)




