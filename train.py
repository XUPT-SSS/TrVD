import argparse
import os

import pandas as pd
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable

def parse_options():
    parser = argparse.ArgumentParser(description='TrVD training.')
    parser.add_argument('-i', '--input', default='mutrvd', choices='mutrvd',
                        help='training dataset type', type=str, required=False)
    parser.add_argument('-m', '--model', default='rvnn-att', choices='rvnn-att',
                        type=str, required=False, help='sub-tree model type')

    parser.add_argument('-d', '--device', default='cuda', choices='cuda, cuda:1, cuda:2',
                        type=str, required=False, help='GPU')
    args = parser.parse_args()
    return args


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item['code'])
        labels.append(item['label'])
    return data, torch.LongTensor(labels)


if __name__ == '__main__':
    args = parse_options()
    embedding_size = 128
    w2v_path = 'subtrees/' + args.input + '/node_w2v_' + str(embedding_size)
    train_data = pd.read_pickle('subtrees/'+args.input+'/train_block.pkl')
    val_data = pd.read_pickle('subtrees/'+args.input+'/dev_block.pkl')

    # filter dataset for code is []
    train_data = train_data.drop(train_data[train_data['code'].str.len() == 0].index)
    val_data = val_data.drop(val_data[val_data['code'].str.len() == 0].index)
    print('train: \n', train_data['label'].value_counts())

    word2vec = Word2Vec.load(w2v_path).wv
    embeddings = np.zeros((word2vec.vectors.shape[0] + 1, word2vec.vectors.shape[1]), dtype="float32")
    embeddings[:word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 86
    EPOCHS = 100
    BATCH_SIZE = 32
    USE_GPU = True
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)
    print('dataset:', args.input)
    model = BatchProgramClassifier(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                                   device, USE_GPU, embeddings)

    if USE_GPU:
        model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.8)
    loss_function = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    model_path = 'saved_model/' + args.input + '/' + args.model
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    best_model = 'saved_model/best_' + args.input + '.pt'

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')
    # training procedure
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        model.train()
        while i < len(train_data):
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels = batch
            if USE_GPU:
                train_inputs, train_labels = train_inputs, train_labels.to(device)
            if len(train_labels) < BATCH_SIZE:
                break
            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs)
            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()
            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item() * len(train_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        print('-' * 89)
        print('| end of epoch {:3d} / {:3d} | time: {:5.2f}s | train loss {:5.2f} | ''train acc {:5.2f} | lr {:.8f}'
              .format(epoch+1, EPOCHS, (time.time() - start_time), total_loss / total, total_acc.item() / total, scheduler.get_lr()[0]))

        if val_data is not None:
            end_time = time.time()
            all_labels = []
            all_preds = []
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            model.eval()
            while i < len(val_data):
                batch = get_batch(val_data, i, BATCH_SIZE)
                i += BATCH_SIZE
                test_inputs, test_labels = batch
                if USE_GPU:
                    test_inputs, test_labels = test_inputs, test_labels.to(device)

                if len(test_labels) < BATCH_SIZE:
                    break

                model.batch_size = len(test_labels)
                output = model(test_inputs)

                loss = loss_function(output, Variable(test_labels))
                _, predicted = torch.max(output.data, 1)
                total_acc += (predicted == test_labels).sum()
                all_labels += test_labels.tolist()
                all_preds += predicted.tolist()

                total += len(test_labels)
                total_loss += loss.item() * len(test_inputs)

            from evaluation import  evaluate_multi
            evaluate_multi(all_preds, all_labels)

            torch.save(model.state_dict(), model_path + '/model_'+str(epoch+1)+'.pt')
            if total_acc.item() / total > best_acc:
                best_acc = total_acc.item() / total
                print('saving best model')
                torch.save(model.state_dict(), best_model)
            print('| end of epoch {:3d} / {:3d} | time: {:5.2f}s | val loss {:5.2f} | val acc {:5.2f}'
                  .format(epoch + 1, EPOCHS, (time.time() - end_time), total_loss / total, total_acc.item() / total))
            print('-' * 89)
            scheduler.step()




