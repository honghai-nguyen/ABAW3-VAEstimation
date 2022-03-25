"""
Author: HuynhVanThong & NguyenHong Hai
Department of AI Convergence, Chonnam Natl. Univ.
"""
import os
import os.path as osp

import torch.nn
from pytorch_lightning import LightningModule
from torchmetrics import F1Score, PearsonCorrCoef, MeanSquaredError
from pl_bolts.optimizers import lr_scheduler as pl_lr_scheduler
from torch.optim import lr_scheduler
from core.metrics import ConCorrCoef
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf
from torch import nn
from torch.nn import functional as F

from core.config import cfg
from core.loss import CCCLoss, CELogitLoss, BCEwithLogitsLoss, MSELoss, SigmoidFocalLoss, CEFocalLoss

from pretrained import vggface2
from pretrained.facex_zoo import get_facex_zoo
from functools import partial
import math
from local_attention import LocalAttention
import numpy as np
import pathlib
import pandas as pd


# Facenet https://github.com/timesler/facenet-pytorch

def get_vggface2(model_name):
    if 'senet50' in model_name:
        vgg2_model = vggface2.resnet50(include_top=False, se=True, num_classes=8631)
        vgg2_ckpt = torch.load('pretrained/vggface2_weights/senet50_ft_weight.pth')
    elif 'resnet50' in model_name:
        vgg2_model = vggface2.resnet50(include_top=False, se=False, num_classes=8631)
        vgg2_ckpt = torch.load('pretrained/vggface2_weights/resnet50_ft_weight.pth')
    else:
        raise ValueError('Unkown model name {} for VGGFACE2'.format(model_name))

    vgg2_model.load_state_dict(vgg2_ckpt['model_state_dict'])
    return vgg2_model


class ABAW3Model(LightningModule):

    def get_backbone(self):
        """
        https://pytorch.org/vision/master/generated/torchvision.models.feature_extraction.create_feature_extractor.html
        :return:
        """
        # TODO: Custom backbone, freeze layers
        backbone_name = self.backbone_name
        if 'regnet' in backbone_name:
            # REGNET - IMAGENET
            regnet_backbone_dict = {'400mf': (regnet_y_400mf, 440), '800mf': (regnet_y_800mf, 784),
                                    '1.6gf': (regnet_y_1_6gf, 888), '3.2gf': (regnet_y_3_2gf, 1512)}

            bb_model = regnet_backbone_dict[backbone_name.split('-')[-1]][0](pretrained=True)
            backbone = create_feature_extractor(bb_model, return_nodes={'flatten': 'feat'})
            # regnet_y: 400mf = 440, 800mf = 784, regnet_y_1_6gf = 888
            # regnet_x: 400mf = 400
            num_feats = {'feat': regnet_backbone_dict[backbone_name.split('-')[-1]][1]}

            if len(self.backbone_freeze) > 0:
                # Freeze backbone model
                for named, param in backbone.named_parameters():
                    do_freeze = True
                    if 'all' not in cfg.MODEL.BACKBONE_FREEZE or not (
                            isinstance(param, nn.BatchNorm2d) and cfg.MODEL.FREEZE_BATCHNORM):
                        for layer_name in self.backbone_freeze:
                            if layer_name in named:
                                do_freeze = False
                                break
                    if do_freeze:
                        param.requires_grad = False

            return backbone, num_feats

        elif backbone_name in ['vggface2-senet50', 'vggface2-resnet50']:
            # PRETRAINED on VGGFACE2
            bb_model = get_vggface2(cfg.MODEL.BACKBONE)
            backbone = create_feature_extractor(bb_model, return_nodes={'flatten': 'feat'})
            num_feats = {'feat': 2048}

            # Freeze backbone model
            for named, param in backbone.named_parameters():
                do_freeze = True
                if 'all' not in cfg.MODEL.BACKBONE_FREEZE or not (
                        isinstance(param, nn.BatchNorm2d) and cfg.MODEL.FREEZE_BATCHNORM):
                    for layer_name in cfg.MODEL.BACKBONE_FREEZE:
                        if layer_name in named:
                            do_freeze = False
                            break
                if do_freeze:
                    param.requires_grad = False

            return backbone, num_feats

        elif backbone_name.split('.')[0] == 'facex':
            # FaceX-Zoo models
            bb_model = get_facex_zoo(backbone_name.split('.')[1],
                                     root_weights=f'{os.getcwd()}/pretrained/facex_zoo')
            # bn is batch norm of Linear layer
            return_node = 'bn' if 'MobileFaceNet' in cfg.MODEL.BACKBONE else 'output_layer'
            backbone = create_feature_extractor(bb_model, return_nodes={return_node: 'feat'})
            num_feats = {'feat': 512}

            if cfg.MODEL.BACKBONE_FREEZE:
                # Freeze backbone model, layer1, layer2, layer3, layer4
                for named, param in backbone.named_parameters():
                    if 'layer9999999' not in named:
                        param.requires_grad = False

            return backbone, num_feats
        elif backbone_name == 'combine':
            # FaceX-Zoo models
            bb_model1 = get_facex_zoo('MobileFaceNet',
                                     root_weights=f'{os.getcwd()}/pretrained/facex_zoo')
            # bn is batch norm of Linear layer
            return_node = 'bn'
            backbone1 = create_feature_extractor(bb_model1, return_nodes={return_node: 'feat'})
            num_feats1 = {'feat': 512}

            if cfg.MODEL.BACKBONE_FREEZE:
                # Freeze backbone model, layer1, layer2, layer3, layer4
                for named, param in backbone1.named_parameters():
                    if 'layer9999999' not in named:
                        param.requires_grad = False

            # REGNET - IMAGENET

            regnet_backbone_dict = {'400mf': (regnet_y_400mf, 440), '800mf': (regnet_y_800mf, 784),
                                    '1.6gf': (regnet_y_1_6gf, 888), '3.2gf': (regnet_y_3_2gf, 1512)}

            bb_model = regnet_backbone_dict['1.6gf'][0](pretrained=True)
            backbone = create_feature_extractor(bb_model, return_nodes={'flatten': 'feat'})
            # regnet_y: 400mf = 440, 800mf = 784, regnet_y_1_6gf = 888
            # regnet_x: 400mf = 400
            num_feats = {'feat': regnet_backbone_dict['1.6gf'][1]}

            if len(self.backbone_freeze) > 0:
                # Freeze backbone model
                for named, param in backbone.named_parameters():
                    do_freeze = True
                    if 'all' not in cfg.MODEL.BACKBONE_FREEZE or not (
                            isinstance(param, nn.BatchNorm2d) and cfg.MODEL.FREEZE_BATCHNORM):
                        for layer_name in self.backbone_freeze:
                            if layer_name in named:
                                do_freeze = False
                                break
                    if do_freeze:
                        param.requires_grad = False
            num_feats = {'feat': num_feats['feat'] + num_feats1['feat']}
            return backbone1, backbone, num_feats
        elif cfg.MODEL.BACKBONE == 'reload':
            backbone = None
            num_feats = {'feat': 10}
            return backbone, num_feats

        else:
            raise ValueError('Only support regnet at this time.')

    def _reset_parameters(self) -> None:
        # Performs ResNet-style weight initialization
        for m_name, m in self.named_modules():
            if 'backbone' in m_name:
                continue
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def __init__(self):
        # TODO: Load backbone pretrained on static frames
        super(ABAW3Model, self).__init__()
        self.seq_len = cfg.DATA_LOADER.SEQ_LEN
        self.task = cfg.TASK
        self.scale_factor = 1.
        self.threshold = 0.5
        self.learning_rate = cfg.OPTIM.BASE_LR
        self.backbone_name = cfg.MODEL.BACKBONE
        self.backbone_freeze = cfg.MODEL.BACKBONE_FREEZE

        if self.task == 'VA':
            # Regression
            self.num_outputs = 2
            self.loss_func = partial(CCCLoss, scale_factor=self.scale_factor)

            self.train_metric = ConCorrCoef(num_classes=self.num_outputs)
            self.val_metric = ConCorrCoef(num_classes=self.num_outputs)

            # 2 x NUM_BINS (e.g., NUM_BINS=10, np.histogram(a, bins=10, range=(-1., 1.))
            # First row: valence, second row: arousal
            self.cls_weights = nn.Parameter(
                torch.tensor([[3.75464546, 1.89663824, 2.57556784, 2.98841786, 3.19120533, 0.27236339,
                               0.43483318, 0.87417645, 1.58016961, 2.36916838],
                              [5.94334483e+02, 4.30892500e+02, 9.76526912e+01, 7.24189076e+01,
                               1.30920623e+01, 2.63744453e-01, 3.75882148e-01, 6.66913017e-01,
                               9.21966354e-01, 1.16343447e+00]], requires_grad=False),
                requires_grad=False) if cfg.TRAIN.LOSS_WEIGHTS else None
        elif self.task == 'EXPR':
            raise ValueError('Do not support EXPR at this time.')
        elif self.task == 'AU':
            raise ValueError('Do not support AU at this time.')
        elif self.task == 'MTL':
            raise ValueError('Do not support MTL at this time.')
        else:
            raise ValueError('Do not know {}'.format(self.task))

        if self.backbone_name == 'combine':
            self.backbone1, self.backbone2, self.num_feats = self.get_backbone()
        else:
            # Dictionary with keys: feats
            self.backbone, self.num_feats = self.get_backbone()

        if 'vggface' in cfg.MODEL.BACKBONE:
            self.down_fc = nn.Sequential(nn.Linear(self.num_feats['feat'], 512), nn.ReLU())
            self.num_feats['feat'] = 512
        else:
            self.down_fc = None

        if cfg.MODEL_TYPE == 'gru':
            self.temporal_module = nn.Sequential(
                nn.GRU(input_size=self.num_feats['feat'],
                       hidden_size=cfg.GRU.HIDDEN_SIZE,
                       num_layers=cfg.GRU.NUM_LAYERS, batch_first=True,
                       dropout=cfg.GRU.DROPOUT,
                       bidirectional=cfg.GRU.BIDIRECTIONAL)
            )
            self.fc = nn.Sequential(nn.Linear((cfg.GRU.HIDDEN_SIZE * (cfg.GRU.BIDIRECTIONAL + 1)), self.num_outputs), nn.Tanh())
            self.local_att = LocalAttention(
                window_size=cfg.GRU.HIDDEN_SIZE * (cfg.GRU.BIDIRECTIONAL + 1),
                causal=True,
                autopad=True  # auto pads both inputs and mask, then truncates output appropriately
            )
        elif cfg.MODEL_TYPE == 'tranf+gru' or cfg.MODEL_TYPE == 'gru+tranf':
            self.gru = nn.Sequential(
                nn.GRU(input_size=self.num_feats['feat'],
                       hidden_size=cfg.GRU.HIDDEN_SIZE,
                       num_layers=cfg.GRU.NUM_LAYERS, batch_first=True,
                       dropout=cfg.GRU.DROPOUT,
                       bidirectional=cfg.GRU.BIDIRECTIONAL)
            )
            # self.fc_gru = nn.Linear(cfg.GRU.HIDDEN_SIZE * (cfg.GRU.BIDIRECTIONAL + 1), self.num_feats['feat'])

            num_enc_dec = cfg.TRANF.NUM_ENC_DEC
            self.tranf = nn.Transformer(d_model=self.num_feats['feat'], nhead=cfg.TRANF.NHEAD,
                                                  num_decoder_layers=num_enc_dec, num_encoder_layers=num_enc_dec,
                                                  dim_feedforward=cfg.TRANF.DIM_FC, dropout=cfg.TRANF.DROPOUT,
                                                  batch_first=True, norm_first=True)
            self.fc_tranf = nn.Linear(self.num_feats['feat'], self.num_outputs)

            self.fc_aux_tranf = nn.Linear(self.num_feats['feat'], self.num_feats['feat'])
            # self.fc_aux_tranf = nn.Linear(self.num_feats['feat'], self.num_outputs)
            self.fc_gru = nn.Linear(cfg.GRU.HIDDEN_SIZE * (cfg.GRU.BIDIRECTIONAL + 1), self.num_outputs)

            self.fc = nn.Linear(self.num_feats['feat'] + cfg.GRU.HIDDEN_SIZE * (cfg.GRU.BIDIRECTIONAL + 1), self.num_outputs)
        else:
            raise ValueError('Do not know {} model'.format(cfg.MODEL_TYPE))
        self._reset_parameters()

        if cfg.MODEL.BACKBONE_PRETRAINED != '':
            print('Load pretrained model on static images')
            pretrained_static = torch.load(cfg.MODEL.BACKBONE_PRETRAINED)['state_dict']
            cur_state_dict = self.state_dict()
            for ky in cur_state_dict.keys():
                if 'backbone' in ky and ky in pretrained_static.keys():
                    cur_state_dict[ky] = pretrained_static[ky]
                # elif 'fc' in ky and ky in pretrained_static.keys():
                #     cur_state_dict[ky] = pretrained_static[ky]
            self.state_dict().update(cur_state_dict)

            pass


    def forward(self, batch):
        if self.backbone is not None:
            if cfg.MODEL.BACKBONE != 'feat+reload':
                image = batch['image']  # batch size x seq x 3 x h x w
                # Convert to batch size * seq x 3 x h x w for feature extraction

                num_seq = image.shape[0]

                feat = torch.reshape(image, (num_seq * self.seq_len,) + image.shape[2:])
                if self.backbone_name == 'combine':
                    feat1 = self.backbone1(feat)['feat']
                    feat2 = self.backbone2(feat)['feat']
                    feat = torch.concat([feat1, feat2], -1)
                else:
                    feat = self.backbone(feat)['feat']  # batch size * seq x num_feat
            else:
                image = batch['image']  # batch size x seq x 3 x h x w
                # Convert to batch size * seq x 3 x h x w for feature extraction

                num_seq = image.shape[0]
                feat = torch.reshape(image, (num_seq * self.seq_len,) + image.shape[2:])
                feat = self.backbone(feat)['feat']  # batch size * seq x num_feat
                feat_reload = batch['feature']

                tmp = feat_reload.reshape((feat_reload.shape[0]*feat_reload.shape[1]*feat_reload.shape[2],
                                           feat_reload.shape[3]))
                feat = torch.cat((feat, tmp), 1)

        else:
            feat = batch['feature']
            num_seq = feat.shape[0]
        # Convert to batch size x seq x num_feat
        feat = torch.reshape(feat, (num_seq, self.seq_len, -1))
        if cfg.MODEL_TYPE == 'gru':
            # GRU
            out_aux = None

            out, _ = self.temporal_module(feat)  # batch size x seq_len x hidden size

            # ######
            out_att = self.local_att(out, out, out)
            out_mean = torch.mean(torch.stack([out, out_att]), dim=0)
            out = self.local_att(out_mean, out_mean, out_mean)
            # #######

        elif cfg.MODEL_TYPE == 'gru+tranf':
            if self.down_fc is not None:
                feat = self.down_fc(feat)
            if self.fc_aux_tranf is not None:
                out_aux = self.fc_aux_tranf(feat)
            else:
                out_aux = None
            # Transfomer
            out_tranf = self.tranf(feat, feat)

            #GRU
            out_gru, _ = self.gru(feat)

            out = torch.cat((out_gru, out_tranf), -1)

        else:
            pass

        if cfg.COMBINE_LOSS:
            out_tranf = self.fc_tranf(out_tranf)
            out = self.fc_gru(out_gru)
            out_stack = torch.stack([out, out_tranf])

            out = out_stack.mean(0)
            return out, out_tranf, out_aux
        else:
            out = self.fc(out)
            return out, out_aux

    def _shared_eval(self, batch, batch_idx, cal_loss=False):
        if cfg.COMBINE_LOSS:
            out, out_tranf, out_aux = self(batch)
        else:
            out, out_aux = self(batch)

        loss = None
        loss_aux_coeff = 0.2
        loss_tranf_coeff = cfg.VALUE_COMBINE_LOSS
        if cal_loss:
            if self.task != 'MTL':
                loss = self.loss_func(out, batch[self.task])
                if out_aux is not None:
                    loss = loss_aux_coeff * self.loss_func(out, batch[self.task]) + (1 - loss_aux_coeff) * loss
                if cfg.COMBINE_LOSS:
                    loss_tranf = self.loss_func(out_tranf, batch[self.task])
                    loss = loss_tranf_coeff * loss_tranf + (1-loss_tranf_coeff) * loss
        return out, loss

    def update_metric(self, out, y, is_train=True):
        if self.task == 'EXPR':
            y = torch.reshape(y, (-1,))
            # out = F.softmax(out, dim=1)
        elif self.task == 'AU':
            out = torch.sigmoid(out)
            y = torch.reshape(y, (-1, self.num_outputs))

        elif self.task == 'VA':
            y = torch.reshape(y, (-1, self.num_outputs))

        out = torch.reshape(out, (-1, self.num_outputs))

        if is_train:
            self.train_metric(out, y)
        else:
            self.val_metric(out, y)

    def training_step(self, batch, batch_idx):

        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=True)

        self.log('train_metric', self.train_metric, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=cfg.TRAIN.BATCH_SIZE)

        return loss

    def validation_step(self, batch, batch_idx):
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=False)

        self.log_dict({'val_metric': self.val_metric, 'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=True,
                      batch_size=cfg.TEST.BATCH_SIZE)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=False)

        if self.task != 'MTL':
            if self.task == 'EXPR':
                out = torch.argmax(F.softmax(out, dim=-1), dim=-1)
            elif self.task == 'AU':
                out = torch.sigmoid(out)

            return out, batch[self.task], batch['index'], batch['video_id']
        else:
            raise ValueError('Do not implement MTL task.')

    def test_step(self, batch, batch_idx):
        # Copy from validation step
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=False)

        self.log_dict({'test_metric': self.val_metric, 'test_loss': loss}, on_step=False, on_epoch=True, prog_bar=True,
                      batch_size=cfg.TEST.BATCH_SIZE)

        return {'preds': out.data.cpu().numpy(),
                'index': batch['index'],
                'video_id': batch['video_id'],
                'VA': batch['VA']}

    def configure_optimizers(self):

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_training_steps = 336 #336

        if cfg.OPTIM.NAME == 'adam':
            print('Adam optimization ', self.learning_rate)
            opt = torch.optim.Adam(model_parameters, lr=self.learning_rate, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        elif cfg.OPTIM.NAME == 'adamw':
            print('AdamW optimization ', self.learning_rate)
            opt = torch.optim.AdamW(model_parameters, lr=self.learning_rate, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        else:
            print('SGD optimization ', self.learning_rate)
            opt = torch.optim.SGD(model_parameters, lr=self.learning_rate, momentum=cfg.OPTIM.MOMENTUM,
                                  dampening=cfg.OPTIM.DAMPENING, weight_decay=cfg.OPTIM.WEIGHT_DECAY)

        opt_lr_dict = {'optimizer': opt}
        lr_policy = cfg.OPTIM.LR_POLICY

        if lr_policy == 'cos':
            warmup_start_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            scheduler = pl_lr_scheduler.LinearWarmupCosineAnnealingLR(opt, warmup_epochs=cfg.OPTIM.WARMUP_EPOCHS,
                                                                      max_epochs=cfg.OPTIM.MAX_EPOCH,
                                                                      warmup_start_lr=warmup_start_lr,
                                                                      eta_min=cfg.OPTIM.MIN_LR)
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'name': 'lr_sched'}})

        elif lr_policy == 'cos-restart':
            min_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            t_0 = cfg.OPTIM.WARMUP_EPOCHS * self.num_training_steps
            print('Number of training steps: ', t_0 // cfg.OPTIM.WARMUP_EPOCHS)
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=t_0, T_mult=cfg.T_MULT,
                                                                 eta_min=min_lr)

            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})

        elif lr_policy == 'cyclic':
            base_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            step_size_up = self.num_training_steps * cfg.OPTIM.WARMUP_EPOCHS // 2
            mode = 'triangular'  # triangular, triangular2, exp_range
            scheduler = lr_scheduler.CyclicLR(opt, base_lr=base_lr, max_lr=self.learning_rate,
                                              step_size_up=step_size_up, mode=mode, gamma=1., cycle_momentum=False)
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})

        elif lr_policy == 'reducelrMetric':
            scheduler = lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=10, min_lr=1e-7, mode='max')
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'name': 'lr_sched',
                                                 "monitor": "val_metric"}})
        else:
            # TODO: add 'exp', 'lin', 'steps' lr scheduler
            pass
        return opt_lr_dict

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        dct = {}
        for i in range(len(outputs)):
            for j in range(len(outputs[i]['video_id'])):
                video_name = outputs[i]['video_id'][j]

                if video_name not in dct.keys():
                    dct[video_name] = {'preds': [], 'index': [], 'VA': []}

                dct[video_name]['preds'].extend(outputs[i]['preds'][j])
                dct[video_name]['index'].extend(outputs[i]['index'][j])
                dct[video_name]['VA'].extend(outputs[i]['VA'][j])

        self.write_results(dct)

    def write_results(self, dct, stage='test'):
        write_folder = self.logger.save_dir

        folder_output = osp.join(write_folder, self.logger.experiment.name, stage)

        if osp.isdir(folder_output):
            num = 1
            folder_output = osp.join(folder_output, '_{}'.format(num))
            while osp.isdir(folder_output):
                num += 1
                folder_output = osp.join(folder_output, '_{}'.format(num))

        if not os.path.exists(folder_output):
            os.mkdir(folder_output)

        print("Folder test: ", folder_output)


        if cfg.TEST_ONLY != 'none':
            # compare length of test
            data_test = np.load(os.path.join(cfg.DATA_LOADER.DATA_DIR, '{}_test.npy'.format(self.task)),
                                allow_pickle=True).item()
        else:
            data_test = {}
        for k in dct.keys():

            if k in data_test.keys():
                len_root = len(data_test[k])
            else:
                len_root = len(dct[k]['preds'])
            file_name = '{}.csv'.format(str(k))
            with open(osp.join(folder_output, file_name), 'a') as fd:
                fd.write('Index,Valence,Arousal,V_Anno,A_Anno\n')
                for i in range(len_root):
                    fd.write('{},{},{},{},{}\n'.format(
                        dct[k]['index'][i],
                        dct[k]['preds'][i][0],
                        dct[k]['preds'][i][1],
                        dct[k]['VA'][i][0],
                        dct[k]['VA'][i][1],))

        self.write_submit(folder_output)

    @staticmethod
    def write_submit(pred_path):
        src_path_pred = pathlib.Path(pred_path)
        dst_path_sub = src_path_pred / 'submit'

        if not os.path.exists(dst_path_sub):
            os.mkdir(dst_path_sub)

        for f in src_path_pred.glob('*csv'):
            basename = os.path.basename(f).replace('.csv', '.txt')

            v = pd.read_csv(f).sort_values(by=['Index']).values
            v_v = v[:, 1]
            v_a = v[:, 2]

            dct = {
                'valence': v_v,
                'arousal': v_a
            }  # 0.4856-w=1

            df = pd.DataFrame(data=dct)
            df.to_csv(os.path.join(dst_path_sub, basename), index=False)
        print("Prediction finish!")
