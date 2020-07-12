import argparse

import cv2
import dataloader
import tqdm
from dataloader import UCF101_splitter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.network import *
from utils.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 motion stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-2, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--demo', dest='demo', action='store_true', help='use model inference on video')


def main():
    global arg
    arg = parser.parse_args()
    print arg

    # Prepare DataLoader
    data_loader = dataloader.Motion_DataLoader(
        BATCH_SIZE=arg.batch_size,
        num_workers=8,
        path='/hdd/UCF-101/Data/optical-flow/',
        ucf_list=os.getcwd() + '/UCF_data_references/',
        ucf_split='01',
        in_channel=10,
    )

    train_loader, test_loader, test_video = data_loader.run()
    # Model
    model = Motion_CNN(
        # Data Loader
        train_loader=train_loader,
        test_loader=test_loader,
        # Utility
        start_epoch=arg.start_epoch,
        resume=arg.resume,
        evaluate=arg.evaluate,
        # Hyper-parameter
        nb_epochs=arg.epochs,
        lr=arg.lr,
        batch_size=arg.batch_size,
        channel=10 * 2,
        test_video=test_video,
        demo=arg.demo
    )
    # Training
    model.run()


class Motion_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, channel,
                 test_video, demo):
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.resume = resume
        self.start_epoch = start_epoch
        self.evaluate = evaluate
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_prec1 = 0
        self.channel = channel
        self.test_video = test_video
        self.demo = demo

    def webcam_inference(self):

        frame_count = 0

        # config the transform to match the network's format
        transform = transforms.Compose([
            transforms.Resize((342, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # prepare the translation dictionary label-action
        data_handler = UCF101_splitter(os.getcwd() + '/UCF_data_references/', None)
        data_handler.get_action_index()
        class_to_idx = data_handler.action_label
        idx_to_class = {v: k for k, v in class_to_idx.iteritems()}

        # Start looping on frames received from webcam
        vs = cv2.VideoCapture(-1)
        softmax = torch.nn.Softmax()
        nn_output = torch.tensor(np.zeros((1, 101)), dtype=torch.float32).cuda()

        while True:
            # read each frame and prepare it for feedforward in nn (resize and type)
            ret, orig_frame = vs.read()
            if ret is False:
                print "Camera disconnected or not recognized by computer"
                break

            frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = transform(frame).view(1, 3, 224, 224).cuda()

            # feed the frame to the neural network
            nn_output += self.model(frame)

            # vote for class with 25 consecutive frames
            if frame_count % 10 == 0:
                nn_output = softmax(nn_output)
                nn_output = nn_output.data.cpu().numpy()
                preds = nn_output.argsort()[0][-5:][::-1]
                pred_classes = [(idx_to_class[str(pred + 1)], nn_output[0, pred]) for pred in preds]

                # reset the process
                nn_output = torch.tensor(np.zeros((1, 101)), dtype=torch.float32).cuda()

            # Display the resulting frame and the classified action
            font = cv2.FONT_HERSHEY_SIMPLEX
            y0, dy = 300, 40
            for i in xrange(5):
                y = y0 + i * dy
                cv2.putText(orig_frame, '{} - {:.2f}'.format(pred_classes[i][0], pred_classes[i][1]),
                            (5, y), font, 1, (0, 0, 255), 2)

            cv2.imshow('frame', orig_frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        vs.release()
        cv2.destroyAllWindows()

    def build_model(self):
        print '==> Build model and setup loss and optimizer'
        self.model = resnet101(pretrained=True, channel=self.channel).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1, verbose=True)

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print "==> loading checkpoint '{}'".format(self.resume)
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print "==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})".format(self.resume, checkpoint['epoch'],
                                                                                     self.best_prec1)
            else:
                print "==> no checkpoint found at '{}'".format(self.resume)
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

        elif self.demo:
            self.model.eval()
            self.webcam_inference()

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True

        if self.evaluate or self.demo:
            return

        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            # lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('../record/motion/motion_video_preds.pickle', 'wb') as f:
                    pickle.dump(self.dic_video_level_preds, f)
                f.close()

            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer': self.optimizer.state_dict()
            }, is_best, 'record/motion/checkpoint.pth.tar', 'record/motion/model_best.pth.tar')

    def train_1epoch(self):
        print '==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to train mode
        self.model.train()
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data, label) in enumerate(progress):
            # measure data loading time
            data_time.update(time.time() - end)

            label = label.cuda(async=True)
            input_var = Variable(data).cuda()
            target_var = Variable(label).cuda()

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
            losses.update(loss.data[0], data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        info = {'Epoch': [self.epoch],
                'Batch Time': [round(batch_time.avg, 3)],
                'Data Time': [round(data_time.avg, 3)],
                'Loss': [round(losses.avg, 5)],
                'Prec@1': [round(top1.avg, 4)],
                'Prec@5': [round(top5.avg, 4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/motion/opf_train.csv', 'train')

    def validate_1epoch(self):
        print '==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs)

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys, data, label) in enumerate(progress):

            # data = data.sub_(127.353346189).div_(14.971742063)
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j].split('-', 1)[0]  # ApplyMakeup_g01_c01
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j, :]
                else:
                    self.dic_video_level_preds[videoName] += preds[j, :]

        # Frame to video level accuracy
        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
        info = {'Epoch': [self.epoch],
                'Batch Time': [round(batch_time.avg, 3)],
                'Loss': [round(video_loss, 5)],
                'Prec@1': [round(video_top1, 3)],
                'Prec@5': [round(video_top5, 3)]
                }
        record_info(info, 'record/motion/opf_test.csv', 'test')
        return video_top1, video_loss

    def frame2_video_level_accuracy(self):

        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds), 101))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii = 0
        for key in sorted(self.dic_video_level_preds.keys()):
            name = key.split('-', 1)[0]

            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name]) - 1

            video_level_preds[ii, :] = preds
            video_level_labels[ii] = label
            ii += 1
            if np.argmax(preds) == (label):
                correct += 1

        # top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()

        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())
        top1, top5 = accuracy(video_level_preds, video_level_labels, topk=(1, 5))

        top1 = float(top1.numpy())
        top5 = float(top5.numpy())

        return top1, top5, loss.data.cpu().numpy()


if __name__ == '__main__':
    main()
