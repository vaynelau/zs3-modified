import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from zs3.modeling.aspp import build_aspp
from zs3.modeling.backbone import build_backbone
from zs3.modeling.decoder import build_decoder
from zs3.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from blocks import *
class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                ),
            )

        for m in self.stages.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = 0
        for stage in self.stages.children():
            h += stage(x)
        return h



class DeepLab(nn.Module):
    def __init__(
        self,
        output_stride=16,
        num_classes=21,
        sync_bn=True,
        freeze_bn=True,
        pretrained=True,
        global_avg_pool_bn=True,
        imagenet_pretrained_path="",
        n_blocks=[3, 4, 23, 3], 
        pyramids=[6, 12, 18, 24],
        hidden = None
    ):
        super().__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
            
        self.add_module(
            "layer1",
            nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)),
                        ("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                    ]
                )
            )
        )
        self.add_module("layer2", _ResBlock(n_blocks[0], 64, 64, 256, 1, 1))
        self.add_module("layer3", _ResBlock(n_blocks[1], 256, 128, 512, 2, 1))
        self.add_module("layer4", _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2))
        self.add_module("layer5", _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4))
        if hidden is not None:
            self.add_module("aspp", _ASPPModule(2048, hidden, pyramids))
        else:
            self.add_module("aspp", _ASPPModule(2048, num_classes, pyramids))
        

#         self.backbone = build_backbone(
#             output_stride,
#             BatchNorm,
#             pretrained=pretrained,
#             imagenet_pretrained_path=imagenet_pretrained_path,
#         )
#         self.aspp = build_aspp(output_stride, BatchNorm, global_avg_pool_bn)
        self.decoder = build_decoder(num_classes, BatchNorm, hidden)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        #x, low_level_feat = self.backbone(input)
        #x = self.aspp(x)
        h = self.layer1(input)
        low_level_feat = self.layer2(h)
        h = self.layer3(low_level_feat)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.aspp(h)

        
        x = self.decoder(h, low_level_feat)
#         x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)
        return x

    def forward_before_class_prediction(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder.forward_before_class_prediction(x, low_level_feat)
        return x

    def forward_class_prediction(self, x, input_size):
        x = self.decoder.forward_class_prediction(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        return x

    def forward_before_last_conv_finetune(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder.forward_before_last_conv_finetune(x, low_level_feat)
        return x

    def forward_class_last_conv_finetune(self, x):
        x = self.decoder.forward_class_last_conv_finetune(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.layer1,self.layer2,self.layer3,self.layer4,self.layer5]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

                            

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class _ConvBatchNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(
                num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
            ),
        )

        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class _Bottleneck(nn.Sequential):
    """Bottleneck Unit"""

    def __init__(
        self, in_channels, mid_channels, out_channels, stride, dilation, downsample
    ):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBatchNormReLU(in_channels, mid_channels, 1, stride, 0, 1)
        self.conv3x3 = _ConvBatchNormReLU(
            mid_channels, mid_channels, 3, 1, dilation, dilation
        )
        self.increase = _ConvBatchNormReLU(
            mid_channels, out_channels, 1, 1, 0, 1, relu=False
        )
        self.downsample = downsample
        if self.downsample:
            self.proj = _ConvBatchNormReLU(
                in_channels, out_channels, 1, stride, 0, 1, relu=False
            )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class _ResBlock(nn.Sequential):
    """Residual Block"""

    def __init__(
        self,
        n_layers,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilation,
        mg=None,
    ):
        super(_ResBlock, self).__init__()

        if mg is None:
            mg = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(mg), "{} values expected, but got: mg={}".format(
                n_layers, mg
            )

        self.add_module(
            "block1",
            _Bottleneck(
                in_channels, mid_channels, out_channels, stride, dilation * mg[0], True
            ),
        )
        for i, g in zip(range(2, n_layers + 1), mg[1:]):
            self.add_module(
                "block" + str(i),
                _Bottleneck(
                    out_channels, mid_channels, out_channels, 1, dilation * g, False
                ),
            )

    def __call__(self, x):
        return super(_ResBlock, self).forward(x)


class _ConvReLU_(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvReLU_, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        '''
        self.add_module(
            "bn",
            nn.BatchNorm2d(
                num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
            ),
        )
        '''
        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvReLU_, self).forward(x)


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True,
                ),
            )

        for m in self.stages.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = 0
        for stage in self.stages.children():
            h += stage(x)
        return h


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none'):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu', dropout='none'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize dropout
        if dropout == 'none':
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False, dropout='none'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize dropout
        if dropout == 'none':
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

