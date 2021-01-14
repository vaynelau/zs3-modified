import os
import os.path as osp
import pathlib

import numpy as np
import scipy
import torch
from PIL import Image
from torchvision import transforms

from zs3.dataloaders import custom_transforms as tr
from .base import BaseDataset, lbl_contains_unseen
from zs3.tools import get_embedding

COCO_DIR = pathlib.Path("./dataset/cocostuff/")


class CocoSegmentation(BaseDataset):
    """
    COCO-stuff dataset
    """

    NUM_CLASSES = 182

    def __init__(
        self,
        args,
        base_dir=COCO_DIR,
        split="train",
        load_embedding=None,
        w2c_size=600,
        weak_label=False,
        unseen_classes_idx_weak=[],
        transform=True,
    ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__(
            args,
            base_dir,
            split,
            load_embedding,
            w2c_size,
            weak_label,
            unseen_classes_idx_weak,
            transform,
        )

        self._image_dir = self._base_dir / "images"
        self._cat_dir = self._base_dir / "annotations"

        self.unseen_classes_idx_weak = unseen_classes_idx_weak

        self.im_ids = []
        self.categories = []

        if self.split == 'train':
            lines = np.load(self._base_dir / 'split/train_list.npy')  # train
        else:
            lines = np.load(self._base_dir / 'split/test_list.npy')  # eval
        lines = [[f.split("/")[-2], f.split("/")[-1].replace(".png", "")] for f in lines]

        seen_classes = torch.from_numpy(np.load(self._base_dir / 'split/seen_cls.npy').astype(np.int32))
        unseen_classes_idx = np.load(self._base_dir / 'split/novel_cls.npy').astype(np.int32)
        novel_classes = torch.from_numpy(np.load(self._base_dir / 'split/novel_cls.npy').astype(np.int32))
        seen_novel_classes = torch.cat((seen_classes, novel_classes), dim=0)

        self.cls_map = torch.tensor([255] * (255 + 1), dtype=torch.float32)
        if len(args.unseen_classes_idx) > 0 and self.split == "train":
            for i, n in enumerate(list(seen_classes)):
                self.cls_map[n] = n
        else:
            for i, n in enumerate(list(seen_novel_classes)):
                self.cls_map[n] = n
        print('self.cls_map', self.cls_map)

        for ii, line in enumerate(lines):
            _image = self._image_dir / line[0] / f"{line[1]}.jpg"
            _cat = self._cat_dir / line[0] / f"{line[1]}.png"
            assert _image.is_file()
            assert _cat.is_file()

            # if unseen classes and training split
            if len(args.unseen_classes_idx) > 0 and self.split == "train" and args.filter_unseen_classes:
                cat = Image.open(_cat)
                cat = np.array(cat, dtype=np.uint8)
                if lbl_contains_unseen(cat, unseen_classes_idx):
                    continue

            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)

        # Display stats
        print(
            "(coco) Number of images in {}: {:d}, {:d} deleted".format(
                split, len(self.images), len(lines) - len(self.images)
            )
        )

    def init_embeddings(self):
        embed_arr = get_embedding(self._base_dir)
        self.make_embeddings(embed_arr)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        if self.weak_label:
            unique_class = np.unique(np.array(_target))
            has_unseen_class = False
            for u_class in unique_class:
                if u_class in self.unseen_classes_idx_weak:
                    has_unseen_class = True
            if has_unseen_class:
                _target = Image.open(
                    "weak_label_context_10_unseen_top_by_image_75.0/pascal/"
                    + self.categories[index].stem
                    + ".jpg"
                )

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split == "train":
                sample = self.transform_tr(sample)
            elif self.split == "val":
                sample = self.transform_val(sample)
        else:
            sample = self.transform_weak(sample)

        if self.load_embedding:
            self.get_embeddings(sample)
        sample["image_name"] = str(self.images[index])
        # print(type(sample["label"]), sample["label"].dtype, sample["label"].size())
        sample["label"] = self.cls_map[sample["label"].long()]
        # print(type(sample["label"]), sample["label"].dtype, sample["label"].size())
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size,
                    crop_size=self.args.crop_size,
                    fill=255,
                ),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose(
            [
#                 tr.FixScale(crop_size=513),
                tr.FixScale(crop_size=self.args.crop_size),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return composed_transforms(sample)

    def transform_weak(self, sample):

        composed_transforms = transforms.Compose(
            [
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def __str__(self):
        return f"COCO(split={self.split})"
