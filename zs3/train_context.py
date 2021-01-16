import os
import json
import numpy as np
import torch
from tqdm import tqdm


from zs3.dataloaders import make_data_loader
from zs3.modeling.deeplab import DeepLab
from zs3.modeling.sync_batchnorm.replicate import patch_replication_callback
from zs3.dataloaders.datasets import DATASETS_DIRS
from zs3.utils.calculate_weights import calculate_weigths_labels
from zs3.utils.loss import SegmentationLosses
from zs3.utils.lr_scheduler import LR_Scheduler
from zs3.utils.metrics import Evaluator
from zs3.utils.saver import Saver
from zs3.utils.summaries import TensorboardSummary
from zs3.parsing import get_parser
from zs3.exp_data import CLASSES_NAMES
from zs3.base_trainer import BaseTrainer, resize_target
from zs3.tools import logWritter, scores_gzsl, get_split, get_config


class Trainer(BaseTrainer):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        """
            Get dataLoader
        """
#         config = get_config(args.config)
#         vals_cls, valu_cls, all_labels, visible_classes, visible_classes_test, train, val, sampler, _, cls_map, cls_map_test = get_split(config)
#         assert (visible_classes_test.shape[0] == config['dis']['out_dim_cls'] - 1)
#         print('seen_classes', vals_cls)
#         print('novel_classes', valu_cls)
#         print('all_labels', all_labels)
#         print('visible_classes', visible_classes)
#         print('visible_classes_test', visible_classes_test)
#         print('train', train[:10], len(train))
#         print('val', val[:10], len(val))
#         print('cls_map', cls_map)
#         print('cls_map_test', cls_map_test)

        # Define Dataloader
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        (self.train_loader, self.val_loader, _, self.nclass,) = make_data_loader(
            args, **kwargs
        )
        print('self.nclass', self.nclass)

        # Define network
        model = DeepLab(
            num_classes=self.nclass,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=False,
            pretrained=args.imagenet_pretrained,
            imagenet_pretrained_path=args.imagenet_pretrained_path,
        )

        train_params = [
            {"params": model.get_1x_lr_params(), "lr": args.lr},
            {"params": model.get_10x_lr_params(), "lr": args.lr * 10},
        ]

        # Define Optimizer
        optimizer = torch.optim.SGD(
            train_params,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = (
                DATASETS_DIRS[args.dataset] / args.dataset + "_classes_weights.npy"
            )
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(
                    args.dataset, self.train_loader, self.nclass
                )
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(
            mode=args.loss_type
        )
        self.model, self.optimizer = model, optimizer
        
        if args.imagenet_pretrained_path is not None:
            state_dict = torch.load(args.imagenet_pretrained_path)
            if 'state_dict' in state_dict.keys():
                self.model.load_state_dict(state_dict['state_dict'])
            else:
                #print(model.state_dict().keys())#['scale.layer1.conv1.conv.weight'])
                #print(state_dict.items().keys())
                new_dict = {}
                for k,v in state_dict.items():
                    #print(k[11:])
                    new_dict[k[11:]] = v
                self.model.load_state_dict(new_dict, strict=False)  # make strict=True to debug if checkpoint is loaded correctly or not if performance is low
                #print(new_dict.keys())
                #print(self.model.state_dict()['layer1.conv1.conv.weight'])
                #model = nn.DataParallel(model, device_ids = [0])
                #self.model.load_state_dict(state_dict, strict=False)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(
            args.lr_scheduler, args.lr, args.epochs, len(self.train_loader)
        )

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            if args.cuda:
                self.model.module.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_pred = checkpoint["best_pred"]
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def validation(self, epoch, args):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0
        targets, outputs = [], []
        log_file = './logs_context_step_1.txt'
        logger = logWritter(log_file)
        for i, sample in enumerate(tbar):
            image, target = sample["image"], sample["label"]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            # print('image', image.size())
            # print('target', target.size())
            # print('output', output.size())
            target = resize_target(target, s=output.size()[2:]).cuda()
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description("Test loss: %.3f" % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy().astype(np.int64)
            pred = np.argmax(pred, axis=1)
            # print('target', target.shape, target.dtype)
            # print('pred', pred.shape, pred.dtype)
            for o, t in zip(pred, target):
                outputs.append(o)
                targets.append(t)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        config = get_config(args.config)
        vals_cls, valu_cls, all_labels, visible_classes, visible_classes_test, train, val, sampler, _, cls_map, cls_map_test = get_split(
            config)
        assert (visible_classes_test.shape[0] == config['dis']['out_dim_cls'] - 1)
        score, class_iou = scores_gzsl(targets, outputs, n_class=len(visible_classes_test),
                                       seen_cls=cls_map_test[vals_cls], unseen_cls=cls_map_test[valu_cls])

        print("Test results:")
        logger.write("Test results:")

        for k, v in score.items():
            print(k + ': ' + json.dumps(v))
            logger.write(k + ': ' + json.dumps(v))

        score["Class IoU"] = {}
        visible_classes_test = sorted(visible_classes_test)
        for i in range(len(visible_classes_test)):
            score["Class IoU"][all_labels[visible_classes_test[i]]] = class_iou[i]
        print("Class IoU: " + json.dumps(score["Class IoU"]))
        logger.write("Class IoU: " + json.dumps(score["Class IoU"]))

        print("Test finished.\n\n")
        logger.write("Test finished.\n\n")

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class, Acc_class_by_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, mIoU_by_class = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar("val/total_loss_epoch", test_loss, epoch)
        self.writer.add_scalar("val/mIoU", mIoU, epoch)
        self.writer.add_scalar("val/Acc", Acc, epoch)
        self.writer.add_scalar("val/Acc_class", Acc_class, epoch)
        self.writer.add_scalar("val/fwIoU", FWIoU, epoch)
        print("Validation:")
        print(
            "[Epoch: %d, numImages: %5d]"
            % (epoch, i * self.args.batch_size + image.data.shape[0])
        )
        print(f"Acc:{Acc}, Acc_class:{Acc_class}, mIoU:{mIoU}, fwIoU: {FWIoU}")
        print(f"Loss: {test_loss:.3f}")

        for i, (class_name, acc_value, mIoU_value) in enumerate(
            zip(CLASSES_NAMES, Acc_class_by_class, mIoU_by_class)
        ):
            self.writer.add_scalar("Acc_by_class/" + class_name, acc_value, epoch)
            self.writer.add_scalar("mIoU_by_class/" + class_name, mIoU_value, epoch)
            print(CLASSES_NAMES[i], "- acc:", acc_value, " mIoU:", mIoU_value)

        new_pred = mIoU
        is_best = False
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        self.saver.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_pred": self.best_pred,
            },
            is_best,
        )


def main():
    parser = get_parser()
    parser.add_argument(
        "--imagenet_pretrained",
        type=bool,
        default=True,
        help="imagenet pretrained backbone",
    )

    parser.add_argument(
        "--out-stride", type=int, default=16, help="network output stride (default: 8)"
    )

    # PASCAL VOC
    parser.add_argument(
        "--dataset",
        type=str,
        default="context",
        choices=["pascal", "coco", "cityscapes", "context"],
        help="dataset name (default: pascal)",
    )

    parser.add_argument("--base-size", type=int, default=312, help="base image size")
    parser.add_argument("--crop-size", type=int, default=312, help="crop image size")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="loss func type (default: ce)",
    )
    # training hyper params

    # PASCAL VOC
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: auto)",
    )

    # PASCAL VOC
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: auto)",
    )
    # checking point
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="put the path to resuming file if needed",
    )
    parser.add_argument(
        "--checkname",
        type=str,
        default="context_4_unseen_filter_unseen_classes_v3",
        help="set the checkpoint name",
    )

    parser.add_argument(
        "--imagenet_pretrained_path",
        type=str,
        default="./checkpoint/deeplabv2_resnet101_init.pth",
        help="set the checkpoint name",
    )

    # evaluation option
    parser.add_argument(
        "--eval-interval", type=int, default=10, help="evaluation interval (default: 1)"
    )

    # 2 unseen
    # unseen_names = ["cow", "motorbike"]
    # 6 unseen
    # unseen_names = ['cow', 'motorbike', 'sofa', 'cat', 'boat', 'fence']
    # 8 unseen
    # unseen_names = ['cow', 'motorbike', 'sofa', 'cat', 'boat', 'fence', 'bird', 'tvmonitor']
    # 10 unseen
    # unseen_names = ['cow', 'motorbike', 'sofa', 'cat', 'boat', 'fence', 'bird', 'tvmonitor', 'aeroplane', 'keyboard']
    # 4 unseen
    unseen_names = ['cow', 'motorbike', 'sofa', 'cat']
    unseen_classes_idx = []
    for name in unseen_names:
        unseen_classes_idx.append(CLASSES_NAMES.index(name))
    # print('unseen_classes_idx', unseen_classes_idx)
    
    parser.add_argument("--unseen_classes_idx", type=int, default=unseen_classes_idx)
    parser.add_argument(
        "--filter_unseen_classes",
        type=bool,
        default=False,
        help="filter unseen classes",
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/context_finetune.yaml',
        help='configuration file for train/val',
    )
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
        except ValueError:
            raise ValueError(
                "Argument --gpu_ids must be a comma-separated list of integers only"
            )

    args.sync_bn = args.cuda and len(args.gpu_ids) > 1

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            "coco": 30,
            "cityscapes": 200,
            "pascal": 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            "coco": 0.1,
            "cityscapes": 0.01,
            "pascal": 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = "deeplab-resnet"
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print("Starting Epoch:", trainer.args.start_epoch)
    print("Total Epoches:", trainer.args.epochs)
    # trainer.validation(trainer.args.start_epoch, args)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (
            args.eval_interval - 1
        ):
            trainer.validation(epoch, args)
    trainer.writer.close()


if __name__ == "__main__":
    main()
