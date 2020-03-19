import random
import torch
import os

from dataloaders.datasets import CaptionDataset
from models.model import BiLSTM_withMaxPooling
import argparse
import json

import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--neg_type', default='standard', type=str)
parser.add_argument('--runname', default='standard', type=str)
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--ex_dup', default=10, type=int)
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--embedding_size', default=512, type=int)
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

net = BiLSTM_withMaxPooling(args.embedding_size, args.embedding_size, device)
model_path = '../trained_models/train_standard_nli_batch_size_32/checkpoint_IC_NLI.pth.tar'
stat_dict = torch.load(model_path, map_location=device)
net = stat_dict['net']
epoch = stat_dict['epoch']
epochs_since_improvement = stat_dict['epochs_since_improvement']
optimizer = stat_dict['optimizer']

data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

if args.run_local:
    net.device = 'cpu'
net.eval()

data_folder = '/home/gal/Desktop/Pycharm_projects/image_captioning/output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

if not args.run_local:
    data_f = '/yoav_stg/gshalev/image_captioning/output_folder'
else:
    data_f = data_folder


# section: word map
word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
print('word_map_file: {}'.format(word_map_file))
print('loading word map from path: {}'.format(word_map_file))
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
print('load word map COMPLETED')
rev_word_map = {v: k for k, v in word_map.items()}

val_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_f, data_name, 'VAL', transform=transforms.Compose([data_normalization])),
    batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

val_caps_lst = torch.load(
        '/yoav_stg/gshalev/image_captioning/output_folder/val_caps_lst' if not args.run_local else os.path.join(
            data_folder, 'val_caps_lst'))

for val_batch_i, (imgs, caps, caplens, all_captions) in enumerate(val_loader):

        # subsection: pos example
    pos_output = net((caps, caplens[0][0]), (all_captions[0][1], caplens[0][1]))
    pred = torch.argmax(pos_output).item()
    sen1 = ' '.join([rev_word_map[x] for x in caps[0].cpu().detach().numpy()[:list(caps[0]).index(0)]][1:-1])
    sen2 = ' '.join([rev_word_map[x] for x in all_captions[0][1].cpu().detach().numpy()[:list(all_captions[0][1]).index(0)]][1:-1])
    print('-------------------')
    print('-------------------')
    print('sen1: {}\nsen2: {}\ngt: {}\npred: {}'.format(sen1, sen2, True, pred))
    print('-------------------')

    fake_caps_and_lens_lst = random.sample(list(val_caps_lst), 1)
    fake_caps_lst = [x[0][0] for x in fake_caps_and_lens_lst]
    fake_caps_lens_lst = [x[1][0] for x in fake_caps_and_lens_lst]
    neg_output = net((caps, caplens[0]), (torch.stack(fake_caps_lst), torch.tensor(fake_caps_lens_lst)))
    pred2 = torch.argmax(neg_output).item()

    sen1 = ' '.join([rev_word_map[x] for x in caps[0].cpu().detach().numpy()[:list(caps[0]).index(0)]][1:-1])
    sen2 = ' '.join([rev_word_map[x] for x in torch.stack(fake_caps_lst)[0].cpu().detach().numpy()[:list(torch.stack(fake_caps_lst)[0]).index(0)]][1:-1])
    print('sen1: {}\nsen2: {}\ngt: {}\npred: {}'.format(sen1, sen2, False, pred2))

    if pred == 1:
        g=0
    if pred2 == 0:
        g=0

l = 0
