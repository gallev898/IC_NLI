import sys


sys.path.append('/home/mlspeech/gshalev/gal/IC_NLI')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

import os
import json
import torch
import random
import argparse

from tqdm import tqdm
from torch import nn
from models.model import BiLSTM_withMaxPooling
from dataloaders.datasets import CaptionDataset
from models.model_image_encoder import Image_encoder

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms


# section: args
parser = argparse.ArgumentParser()
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--neg_type', default='standard', type=str)
parser.add_argument('--runname', default='standard', type=str)
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--embedding_size', default=512, type=int)
args = parser.parse_args()

# section: W&B
if not args.run_local:
    import wandb


    wandb.init(project="IC_NLI", name=args.runname, dir='/yoav_stg/gshalev/wandb')
    # wandb login a8c4526db3e8aa11d7b2674d7c257c58313b45ca

# section: initialization
device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_folder = '/home/gal/Desktop/Pycharm_projects/image_captioning/output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

if not args.neg_type == 'standard':
    df = '/yoav_stg/gshalev/image_captioning/output_folder'
    if args.run_local:
        replace_dic_load_path = '{}_masking_train_caps'.format(args.neg_type)
        noun_idx_set = torch.load('noun_idx_set')
        verb_idx_set = torch.load('verb_idx_set')
    else:
        replace_dic_load_path = os.path.join(df, '{}_masking_train_caps'.format(args.neg_type))
        noun_idx_set = torch.load('/yoav_stg/gshalev/image_captioning/output_folder/noun_idx_set')
        verb_idx_set = torch.load('/yoav_stg/gshalev/image_captioning/output_folder/verb_idx_set')


def calc_batch_accuracy(out, label):
    batch_accuracy = 0
    y_tags = [torch.argmax(o).item() for o in out]

    for y, y_tag in zip(label, y_tags):
        if y_tag == y:
            batch_accuracy += 1

    return batch_accuracy



def get_fake(args_, caps_lst_):
    fake_caps_and_lens_lst = random.sample(list(caps_lst_), args_.batch_size)
    fake_caps_lst = [x[0][0] for x in fake_caps_and_lens_lst]
    fake_caps_lens_lst = [x[1][0] for x in fake_caps_and_lens_lst]

    fake_caps = torch.stack(fake_caps_lst)
    fake_caps_lens = torch.stack(fake_caps_lens_lst)
    return fake_caps, fake_caps_lens


def create_standard_examples(args_, image_encoder_, imgs_, caps):
    images = []
    samples = []
    for img, c in zip(imgs_, caps):
        images.append(img.to(device))
        samples.append(random.sample(list(c), 1)[0])
    # encoded_imgs = image_encoder_(torch.stack(images).to(device))
    ex = list(zip(images, torch.stack(samples)))
    return random.sample(ex, int(args_.batch_size / 2))


def create_standard_neg_examples(args_, image_encoder_, imgs_, caps):

    images = []
    samples = []
    for img, c in zip(imgs_, caps):
        images.append(img.to(device))
        samples.append(c)
    # encoded_imgs = image_encoder_(torch.stack(images).to(device))
    ex = list(zip(images, samples))
    # ex = list(zip(encoded_imgs, samples))
    return random.sample(ex, int(args_.batch_size / 2))


def train(args_, train_loader_, train_caps_lst_, optimizer_, net_, criterion_, image_encoder_):
    print('Sterting train')
    train_batch_loss = []
    train_batch_accuracy = []

    for train_batch_i, (imgs, caps, caplens, all_captions) in enumerate(tqdm(train_loader_)):
        if imgs.shape[0] != args.batch_size:
            print('skipped batch # {}'.format(
                train_batch_i))  # notice: to prevent 'index out of bound' in 'create_standard_neg_examples'
            continue

        # subsection: create pos examples
        pos_examples = create_standard_examples(args, image_encoder, imgs, all_captions)

        # subsection: create neg examples
        fake_caps, fake_caps_lens = get_fake(args_, train_caps_lst_)

        if args_.neg_type == 'standard':
            neg_ex = create_standard_neg_examples(args_, image_encoder_, imgs, fake_caps)

        if args.neg_type == 'replae_verb':
            temp = 0

        if train_batch_i == 0:
            print('#imgs: {}'.format(imgs.shape[0]))
            print('#pos examples: {}'.format(len(pos_examples)))
            print('#neg examples: {}'.format(len(neg_ex)))

        # subsection: Learn
        optimizer_.zero_grad()
        examples = pos_examples + neg_ex

        seq = torch.stack([x[1] for x in examples]).to(device)
        u = torch.stack([x[0] for x in examples]).squeeze(1)
        u = image_encoder_(u)
        v = net_.encode((seq, torch.zeros(seq.shape)))

        representation = torch.cat((u, v, torch.abs(u - v), u * v), 1)

        out = net_.fc1(representation)
        out = torch.tanh(out)
        outputs = net_.fc2(out)

        labels = torch.cat(
            (torch.zeros(len(pos_examples), dtype=torch.long), torch.ones(len(neg_ex), dtype=torch.long))).to(
            device)
        loss = criterion_(outputs, labels)
        loss.backward()
        optimizer_.step()

        # subsection: metrics
        batch_accuracy = calc_batch_accuracy(outputs, labels)

        train_batch_accuracy.append(batch_accuracy / float(len(outputs)))
        train_batch_loss.append(loss.item())

    return np.average(train_batch_accuracy), np.average(train_batch_loss)


def validation(args_, val_loader_, optimizer_, val_caps_lst_, net_, criterion_, image_encoder_):
    print('Starting val')
    dev_batch_loss = []
    dev_batch_accuracy = []

    with torch.no_grad():
        for val_batch_i, (imgs, caps, caplens, all_captions) in enumerate(tqdm(val_loader_)):
            dev_accuracy = 0

            if args_.neg_type == 'standard':
                optimizer_.zero_grad()

                fake_caps_and_lens_lst = random.sample(list(val_caps_lst_), 1)
                fake_caps_lst = [x[0][0] for x in fake_caps_and_lens_lst]

                # subsection: pos example
                u = image_encoder_(imgs.to(device))
                v = net_.encode((all_captions[0][1].to(device), 0))

                representation = torch.cat((u, v, torch.abs(u - v), u * v), 1)

                out = net_.fc1(representation)
                out = torch.tanh(out)
                pos_output = net_.fc2(out)

                # subsection: neg example
                v = net_.encode((torch.stack(fake_caps_lst), 0))
                representation = torch.cat((u, v, torch.abs(u - v), u * v), 1)

                out = net_.fc1(representation)
                out = torch.tanh(out)
                neg_output = net_.fc2(out)

                if torch.argmax(pos_output).item() == 0:
                    dev_accuracy += 1
                if torch.argmax(neg_output).item() == 1:
                    dev_accuracy += 1

                loss = criterion_(pos_output, torch.zeros(1, dtype=torch.long).to(device))
                loss2 = criterion_(neg_output, torch.ones(1, dtype=torch.long).to(device))

                dev_batch_loss.append((loss.item() + loss2.item()) / float(2))
                dev_batch_accuracy.append(dev_accuracy / float(2))

    return np.average(dev_batch_accuracy), np.average(dev_batch_loss)


if __name__ == '__main__':
    # section: data paths
    if not args.run_local:
        data_f = '/yoav_stg/gshalev/image_captioning/output_folder'
    else:
        data_f = data_folder

    # section: save dir
    if not args.run_local:
        if not os.path.exists("/yoav_stg/gshalev/IC_NLI/trained_models/{}".format(args.runname)):
            os.mkdir("/yoav_stg/gshalev/IC_NLI/trained_models/{}".format(args.runname))

    # section: word map
    word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
    print('word_map_file: {}'.format(word_map_file))
    print('loading word map from path: {}'.format(word_map_file))
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    print('load word map COMPLETED')

    # section: data loaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_f, data_name, 'TRAIN', transform=transforms.Compose([data_normalization])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)  # NOTICE: shuffle=False

    if not args.debug:
        val_loader = torch.utils.data.DataLoader(
            CaptionDataset(data_f, data_name, 'VAL', transform=transforms.Compose([data_normalization])),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    else:
        val_loader = None

    # section: caps lsts
    train_caps_lst = torch.load(
        '/yoav_stg/gshalev/image_captioning/output_folder/train_caps_lst' if not args.run_local else os.path.join(
            data_folder, 'train_caps_lst'))
    val_caps_lst = torch.load(
        '/yoav_stg/gshalev/image_captioning/output_folder/val_caps_lst' if not args.run_local else os.path.join(
            data_folder, 'val_caps_lst'))

    # section: load trained NLI
    model_path = '../trained_models/train_standard_nli_batch_size_32/checkpoint_IC_NLI.pth.tar' if args.run_local \
        else '/yoav_stg/gshalev/IC_NLI/trained_models/train_standard_nli_batch_size_32/checkpoint_IC_NLI.pth.tar'
    stat_dict = torch.load(model_path, map_location=device)
    net = stat_dict['net']
    #notice
    for param in net.parameters():
        param.requires_grad = False
    net.to(device)

    # section: image encoder instance
    image_encoder = Image_encoder(device, 1024)
    image_encoder.to(device)

    # section: wandb
    if not args.run_local:
        wandb.watch(net)

    # section: optimizer and critrin
    optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, image_encoder.parameters()), lr=LEARNING_RATE, )
    criterion = nn.CrossEntropyLoss()

    # section: params
    last_dev_accuracy = 0
    last_dev_epoch_running_loss = np.inf
    epoch_losses = []
    epoch_accuracies = []
    epoch_number = 0
    last_val_accuracy = 0
    epochs_since_improvement = 0

    # section: start epoch
    while epochs_since_improvement < 6:
        epoch_number += 1
        print('******************STARTING EPOCH NUM: {} , LR: {}******************'.format(epoch_number, LEARNING_RATE))

        # section: training
        train_accuracy, train_loss = train(args, train_loader, train_caps_lst, optimizer, net, criterion, image_encoder)

        # section: val
        val_accuracy, val_loss = validation(args, val_loader, optimizer, val_caps_lst, net, criterion, image_encoder)

        # section: adjust LR
        if last_val_accuracy == 0:
            last_val_accuracy = val_accuracy

        if val_accuracy <= last_val_accuracy:
            epochs_since_improvement += 1
            LEARNING_RATE /= 5
            optimizer.param_groups[0]['lr'] = LEARNING_RATE
        else:
            epochs_since_improvement = 0
            state = {'epoch': epoch_number,
                     'epochs_since_improvement': epochs_since_improvement,
                     'image_encoder': image_encoder,
                     'optimizer': optimizer,
                     }
            filename = 'checkpoint_IC_NLI.pth.tar'
            torch.save(state, os.path.join('/yoav_stg/gshalev/IC_NLI/trained_models/{}'.format(args.runname), filename))
            print('Saved madel to: {}'.format(
                os.path.join('/yoav_stg/gshalev/IC_NLI/trained_models/{}'.format(args.runname), filename)))

        last_val_accuracy = val_accuracy

        print('Train epochs running accuracy: {}'.format(train_accuracy))
        print('Train epochs running loss {}'.format(train_loss))
        print('Dev epoch accuracy: {}'.format(val_accuracy))
        print('Dev epoch loss: {}'.format(val_loss))

        if not args.run_local:
            wandb.log({"Train Accuracy": train_accuracy,
                       "Train Loss": train_loss,
                       'Dev Accuracy': val_accuracy,
                       'Dev Loss': val_loss})
# train_image_encoder_nli.py
