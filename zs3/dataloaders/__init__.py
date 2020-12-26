from torch.utils.data import DataLoader
import os
import numpy as np
import torch

from zs3.dataloaders.datasets import combine_dbs, pascal, sbd, context

from .cocostuff import CocoStuff10k, CocoStuff164k, LoaderZLS


def get_dataset(name):
    return {"cocostuff10k": CocoStuff10k, "cocostuff164k": CocoStuff164k, "LoaderZLS": LoaderZLS}[name]


class RandomImageSampler(torch.utils.data.Sampler):
    """
    Samples classes randomly, then returns images corresponding to those classes.
    """

    def __init__(self, seenset, novelset):
        self.data_index = []
        for v in seenset:
            self.data_index.append([v, 0])
        for v, i in novelset:
            self.data_index.append([v, i+1])

    def __iter__(self):
        return iter([self.data_index[i] for i in np.random.permutation(len(self.data_index))])

    def __len__(self):
        return len(self.data_index)


def get_split(cfg):
    dataset_path = os.path.join(cfg['datadir'], cfg['dataset'])
    train = np.load(dataset_path + '/split/train_list.npy')
    val = np.load(dataset_path + '/split/test_list.npy')

    seen_classes = np.load(dataset_path + '/split/seen_cls.npy').astype(np.int32)
    novel_classes = np.load(dataset_path + '/split/novel_cls.npy').astype(np.int32)
    seen_novel_classes = np.concatenate((seen_classes, novel_classes), axis=0)
    all_labels = np.genfromtxt(dataset_path + '/labels_refined.txt', delimiter='\t', usecols=1, dtype='str')

    visible_classes = seen_classes
    visible_classes_test = seen_novel_classes

    novelset, seenset = [], range(train.shape[0])
    sampler = RandomImageSampler(seenset, novelset)

    cls_map = np.array([0]*256).astype(np.int32)
    for n in list(seen_classes):
        cls_map[n] = n
    cls_map[255] = 255
    cls_map_test = np.array([cfg['ignore_index']]*(cfg['ignore_index']+1)).astype(np.int32)
    for i, n in enumerate(list(seen_novel_classes)):
        cls_map_test[n] = i

    visibility_mask = {}
    visibility_mask[0] = cls_map.copy()
    for i, n in enumerate(list(novel_classes)):
        visibility_mask[i+1] = cls_map.copy()
        visibility_mask[i+1][n] = seen_classes.shape[0] + i + 1
#     print('seen_classes', seen_classes)
#     print('novel_classes', novel_classes)
#     print('all_labels', all_labels)
#     print('visible_classes', visible_classes)
#     print('visible_classes_test', visible_classes_test)
#     print('visibility_mask', visibility_mask)
#     print('train', train[:10], len(train))
#     print('val', val[:10], len(val))
    
    return seen_classes, novel_classes, all_labels, visible_classes, visible_classes_test, train, val, sampler, visibility_mask, cls_map, cls_map_test


def make_data_loader(
    args,
    transform=True,
    load_embedding=None,
    w2c_size=300,
    weak_label=False,
    unseen_classes_idx_weak=[],
    **kwargs,
):
    if args.dataset == "pascal":
        train_set = pascal.VOCSegmentation(
            args,
            transform=transform,
            split="train",
            load_embedding=load_embedding,
            w2c_size=w2c_size,
            weak_label=weak_label,
            unseen_classes_idx_weak=unseen_classes_idx_weak,
        )
        val_set = pascal.VOCSegmentation(
            args, split="val", load_embedding=load_embedding, w2c_size=w2c_size
        )
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(
                args,
                transform=transform,
                split=["train_noval"],
                load_embedding=load_embedding,
                w2c_size=w2c_size,
                weak_label=weak_label,
                unseen_classes_idx_weak=unseen_classes_idx_weak,
            )
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoader(
            val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )
        test_loader = None
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == "context":
        train_set = context.ContextSegmentation(
            args,
            transform=transform,
            split="train",
            load_embedding=load_embedding,
            w2c_size=w2c_size,
            weak_label=weak_label,
            unseen_classes_idx_weak=unseen_classes_idx_weak,
        )
        val_set = context.ContextSegmentation(
            args, split="val", load_embedding=load_embedding, w2c_size=w2c_size
        )
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        val_loader = DataLoader(
            val_set, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError
