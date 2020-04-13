"""
 Between-class Learning for Image Classification.
 Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada

"""

import sys
import os
import chainer

import opts
import models
import dataset
from train import Trainer
import numpy as np
import matplotlib.pyplot as plt


def main():
    opt = opts.parse()
    chainer.cuda.get_device_from_id(opt.gpu).use()
    for i in range(1, opt.nTrials + 1):
        print('+-- Trial {} --+'.format(i))
        train(opt, i)


def train(opt, trial):
    model = getattr(models, opt.netType)(opt.nClasses)
    model.to_gpu()
    optimizer = chainer.optimizers.NesterovAG(lr=opt.LR, momentum=opt.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(opt.weightDecay))
    train_iter, val_iter = dataset.setup(opt)
    trainer = Trainer(model, optimizer, train_iter, val_iter, opt)
    
    train_error = np.empty(opt.nEpochs)
    test_error = np.empty(opt.nEpochs)

    for epoch in range(1, opt.nEpochs + 1):
        train_loss, train_top1 = trainer.train(epoch)
        val_top1 = trainer.val()
        sys.stderr.write('\r\033[K')
        sys.stdout.write(
            '| Epoch: {}/{} | Train: LR {}  Loss {:.3f}  top1 {:.2f} | Val: top1 {:.2f}\n'.format(
                epoch, opt.nEpochs, trainer.optimizer.lr, train_loss, train_top1, val_top1))
        sys.stdout.flush()
        
        train_error[epoch-1] = train_top1
        test_error[epoch-1] = val_top1

    if opt.save != 'None':
        chainer.serializers.save_npz(
            os.path.join(opt.save, 'model_trial{}.npz'.format(trial)), model)
    
    plt.figure()    
    plt.plot(np.arange(opt.nEpochs),np.array(train_error))
    plt.plot(np.arange(opt.nEpochs),np.array(test_error))
    plt.ylim([0,30])
    plt.xlabel('Epoch')
    plt.ylabel('Test Error (%)')
    plt.legend((['Train error','Test error']))
    axes = plt.gca()
    axes.yaxis.grid()
    plt.savefig('.'+'/results'+'/trial'+str(trial)+'.jpg',format='jpg')
    plt.close()
    
    

if __name__ == '__main__':
    main()
