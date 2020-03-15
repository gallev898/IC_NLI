import sys


sys.path.append('/home/mlspeech/gshalev/gal/IC_NLI')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

from tqdm import tqdm
from dataloaders.datasets import CaptionDataset
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn


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

# section: W&B
if not args.run_local:
    # wandb login a8c4526db3e8aa11d7b2674d7c257c58313b45ca
    import wandb


    wandb.init(project="IC_NLI", name=args.runname, dir='/yoav_stg/gshalev/wandb')

data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data_folder = '/home/gal/Desktop/Pycharm_projects/image_captioning/output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

if not args.neg_type == 'standard':
    replace_dic_load_path = '/yoav_stg/gshalev/image_captioning/output_folder/{}_masking_train_caps'.format(
        args.neg_type) if not args.run_local else '{}_masking_train_caps'.format(args.neg_type)
    replace_dic = torch.load(replace_dic_load_path)
    print('successfully loades *replace_dic* from: {}'.format(replace_dic_load_path))
    noun_idx_set = torch.load(
        'noun_idx_set' if args.run_local else '/yoav_stg/gshalev/image_captioning/output_folder/noun_idx_set')
    print('successfully loades *noun_idx_set*')
    verb_idx_set = torch.load(
        'verb_idx_set' if args.run_local else '/yoav_stg/gshalev/image_captioning/output_folder/verb_idx_set')
    print('successfully loades *verb_idx_set*')


class BiLSTM_withMaxPooling(nn.Module):
    device = None


    def __init__(self, lstm_input_size, lstm_hidden_zise, run_device):
        super(BiLSTM_withMaxPooling, self).__init__()
        self.device = run_device
        self.bilstm = nn.LSTM(lstm_input_size, lstm_hidden_zise, bidirectional=True)
        self.fc1 = nn.Linear(lstm_hidden_zise * 8, 512)
        self.fc2 = nn.Linear(512, 2)
        self.embedding = nn.Embedding(9490, 512)  # embedding layer


    def forward(self, sent1, sent2):
        u = self.encode(sent1)
        v = self.encode(sent2)

        representation = torch.cat((u, v, torch.abs(u - v), u * v), 1)

        out = self.fc1(representation)
        out = torch.tanh(out)
        out = self.fc2(out)
        return out


    def encode(self, sent):
        input, seq_len = sent
        input = input.to(device)
        input = self.embedding(input)
        if len(input.shape) == 3:
            input = input.permute(1, 0, 2)  # seq_len, batch, input_size
        else:
            input = input.unsqueeze(1)
        input = input.to(self.device)
        output = self.bilstm(input)[0]
        max_pooling = torch.max(output, 0)[0].to(device)
        return max_pooling


def calc_batch_accuracy(out, label):
    batch_accuracy = 0
    y_tags = [torch.argmax(o).item() for o in out]

    for y, y_tag in zip(label, y_tags):
        if y_tag == y:
            batch_accuracy += 1

    return batch_accuracy


def create_pos_examples(args_, all_captions_, caplens_):
    pos_ex = []

    assert args_.ex_dup in [10, 6, 3, 2]
    # todo: sampling configuration
    ex_map = {10: 5, 6: 4, 3: 3, 2: 2}
    num_of_examples = ex_map[args.ex_dup]

    for b, l in zip(all_captions_, caplens_):
        for i in range(len(b)):
            j = i + 1
            # while j < 2:
            while j < num_of_examples:
                pos_ex.append((b[i], l[i], b[j], l[j]))
                j += 1

    # return pos_ex
    return random.sample(pos_ex, int(args_.batch_size / 2))


def get_fake(args_, caps_lst_):
    fake_caps_and_lens_lst = random.sample(list(caps_lst_), (args_.batch_size) * 10)
    fake_caps_lst = [x[0][0] for x in fake_caps_and_lens_lst]
    fake_caps_lens_lst = [x[1][0] for x in fake_caps_and_lens_lst]

    fake_caps = torch.stack(fake_caps_lst)
    fake_caps_lens = torch.stack(fake_caps_lens_lst)
    return fake_caps, fake_caps_lens


def create_standard_neg_examples(args_, all_captions_, caplens_, fake_caps_, fake_caps_lens_):
    neg_ex = []
    # s1, l1, s2 l2
    j = 0
    while j < args_.batch_size * args_.ex_dup:
        for b, l in zip(all_captions_, caplens_):
            for i in range(len(b)):
                neg_ex.append(
                    (b[i], l[i], fake_caps_[j], fake_caps_lens_[j]))  # NOTICE: good only for train_loader shuffle=true
                j += 1
    return random.sample(neg_ex, int(args_.batch_size / float(2)))
    # return neg_ex


def train(args_, train_loader_, train_caps_lst_, optimizer_, net_, criterion_):
    print('Sterting train')
    train_batch_loss = []
    train_batch_accuracy = []

    for train_batch_i, (imgs, caps, caplens, all_captions) in enumerate(tqdm(train_loader_)):
        if imgs.shape[0] != args.batch_size:
            print('skipped batch # {}'.format(
                train_batch_i))  # notice: to prevent 'index out of bound' in 'create_standard_neg_examples'
            continue

        # subsection: create pos examples
        pos_examples = create_pos_examples(args_, all_captions, caplens)

        # subsection: create neg examples
        fake_caps, fake_caps_lens = get_fake(args_, train_caps_lst_)

        if args_.neg_type == 'standard':
            neg_ex = create_standard_neg_examples(args_, all_captions, caplens, fake_caps, fake_caps_lens)

        if args.neg_type == 'replae_verb':
            caps_key = []
            # for b, l in zip(all_captions, caplens):
            #     caps_key += [str(c).replace('\n', '') for c in b]
            #
            # for k in caps_key:
            #     fake_caps_lst.append(replace_dic[k][0].squeeze(0))
            #     fake_caps_lens_lst.append(replace_dic[k][1].squeeze(0))
            # fake_zero_vectors.append(replace_dic[k][2])
            # pos_indexes.append(replace_dic[k][3])
            # fake_caps_lst = fake_caps_lst[:num_of_fake]
            # fake_caps_lens_lst = fake_caps_lens_lst[:num_of_fake]
            # fake_zero_vectors = fake_zero_vectors[:num_of_fake]
            # pos_indexes = pos_indexes[:num_of_fake]
            #
            # fake_caps_lens = sum((torch.stack(fake_caps_lens_lst) - 1).squeeze(1)).item()
            #
            # if args.neg_type in ['noun', 'verb']:
            #     for fake, pos_idx in zip(fake_caps_lst, pos_indexes):
            #         random_samples = random.sample(
            #             list(noun_idx_set if args.neg_type == 'noun' else verb_idx_set),
            #             len(pos_idx))
            #         for i, pi in enumerate(pos_idx):
            #             fake[pi] = random_samples[i]
            # else:  # == full
            #     for fake, pos_idx in zip(fake_caps_lst, pos_indexes):
            #         noun_idx = pos_idx[0]
            #         verb_idx = pos_idx[1]
            #         random_noun_samples = random.sample(list(noun_idx_set), len(noun_idx))
            #         random_verb_samples = random.sample(list(verb_idx_set), len(verb_idx))
            #         for i, ni in enumerate(noun_idx):
            #             fake[ni] = random_noun_samples[i]
            #         for i, vi in enumerate(verb_idx):
            #             fake[vi] = random_verb_samples[i]

        if train_batch_i == 0:
            print('#imgs: {}'.format(imgs.shape[0]))
            print('#pos examples: {}'.format(len(pos_examples)))
            print('#neg examples: {}'.format(len(neg_ex)))

        # subsection: Learn
        optimizer_.zero_grad()
        examples = pos_examples + neg_ex
        input1 = torch.stack([x[0] for x in examples])
        input1_lens = [x[1] for x in examples]
        input2 = torch.stack([x[2] for x in examples])
        input2_lens = [x[3] for x in examples]

        outputs = net_((input1, input1_lens), (input2, input2_lens))

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


def validation(args_, val_loader_, optimizer_, val_caps_lst_, net_, criterion_):
    print('Starting val')
    dev_batch_loss = []
    dev_batch_accuracy = []

    with torch.no_grad():
        # dev_epoch_runing_loss = 0
        for val_batch_i, (imgs, caps, caplens, all_captions) in enumerate(tqdm(val_loader_)):
            dev_accuracy = 0

            if args_.neg_type == 'standard':
                optimizer_.zero_grad()

                fake_caps_and_lens_lst = random.sample(list(val_caps_lst_), 1)
                fake_caps_lst = [x[0][0] for x in fake_caps_and_lens_lst]
                fake_caps_lens_lst = [x[1][0] for x in fake_caps_and_lens_lst]

                # subsection: pos example
                pos_output = net_((caps, caplens[0][0]), (all_captions[0][1], caplens[0][1]))
                # pos_output = net((caps, caplens[0][0]), (all_captions[0][1], caplens[0][1]))
                # subsection: neg example
                neg_output = net_((caps, caplens[0]), (torch.stack(fake_caps_lst), torch.tensor(fake_caps_lens_lst)))

                if torch.argmax(pos_output).item() == 0:
                    dev_accuracy += 1
                if torch.argmax(neg_output).item() == 1:
                    dev_accuracy += 1

                loss = criterion_(pos_output, torch.zeros(1, dtype=torch.long).to(device))
                loss2 = criterion_(neg_output, torch.ones(1, dtype=torch.long).to(device))

                dev_batch_loss.append((loss.item() + loss2.item()) / float(2))
                dev_batch_accuracy.append(dev_accuracy / float(2))
                # dev_accuracy = 0
        #
        # if dev_epoch_runing_loss >= last_dev_epoch_running_loss:
        #     LEARNING_RATE /= 5
        #     optimizer.param_groups[0]['lr'] = LEARNING_RATE
        # else:
        #     LEARNING_RATE *= 0.99
        #     optimizer.param_groups[0]['lr'] = LEARNING_RATE
        #
        # last_dev_epoch_running_loss = dev_epoch_runing_loss
        #
        # train_accuracy = epoch_running_accuracy / float(train_batch_i)
        # train_loss = epoch_running_loss / float(train_batch_i)
        # dev_accuracy = np.average(dev_batch_accuracy)
        # dev_loss = np.average(dev_batch_loss)
        # print('Train epochs running accuracy: {}'.format(train_accuracy))
        # print('Train epochs running loss {}'.format(train_loss))
        # print('Dev epoch accuracy: {}'.format(dev_accuracy))
        # print('Dev epoch loss: {}'.format(dev_loss))
        #
        # if not args.run_local:
        #     wandb.log({"Train Accuracy": train_accuracy,
        #                "Train Loss": train_loss,
        #                'Dev Accuracy': dev_accuracy,
        #                'Dev Loss': dev_loss})

    return np.average(dev_batch_accuracy), np.average(dev_batch_loss)


if __name__ == '__main__':
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 5e-4

    # section: paths
    if not args.run_local:
        data_f = '/yoav_stg/gshalev/image_captioning/output_folder'
    else:
        data_f = data_folder

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
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)  # NOTICE

    train_caps_lst = torch.load(
        '/yoav_stg/gshalev/image_captioning/output_folder/train_caps_lst' if not args.run_local else os.path.join(
            data_folder, 'train_caps_lst'))
    val_caps_lst = torch.load(
        '/yoav_stg/gshalev/image_captioning/output_folder/val_caps_lst' if not args.run_local else os.path.join(
            data_folder, 'val_caps_lst'))

    if not args.debug:
        val_loader = torch.utils.data.DataLoader(
            CaptionDataset(data_f, data_name, 'VAL', transform=transforms.Compose([data_normalization])),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
    else:
        val_loader = None

    # section: initializations
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

    net = BiLSTM_withMaxPooling(args.embedding_size, args.embedding_size, device)

    net.to(device)

    # section: wandb
    if not args.run_local:
        wandb.watch(net)

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    last_dev_accuracy = 0
    last_dev_epoch_running_loss = np.inf
    epoch_losses = []
    epoch_accuracies = []
    epoch_number = 0
    last_val_accuracy = 0
    epochs_since_improvement = 0

    while epochs_since_improvement < 6:
        epoch_number += 1
        print('******************STARTING EPOCH NUM: {} , LR: {}******************'.format(epoch_number, LEARNING_RATE))

        # section: training
        train_accuracy, train_loss = train(args, train_loader, train_caps_lst, optimizer, net, criterion)

        # section: val
        val_accuracy, val_loss = validation(args, val_loader, optimizer, val_caps_lst, net, criterion)

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
                     'net': net,
                     'optimizer': optimizer,
                     }
            filename = 'checkpoint_IC_NLI.pth.tar'
            torch.save(state, os.path.join('/yoav_stg/gshalev/IC_NLI/trained_models/{}'.format(args.runname), filename))
            print('Saved madel to: {}'.format(os.path.join('/yoav_stg/gshalev/IC_NLI/trained_models/{}'.format(args.runname), filename)))

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
# train_nli.py
