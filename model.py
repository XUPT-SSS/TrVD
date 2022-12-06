import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze(-1)

def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu,
                 device, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_l = nn.Linear(encode_dim, encode_dim)
        self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        self.device = device
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

        self.agg_net = torch.nn.GRU(embedding_dim, encode_dim, 1)
        self.W_r = nn.Linear(encode_dim, encode_dim)

        # query attention
        self.sent_weight = nn.Parameter(torch.Tensor(encode_dim, encode_dim))
        self.sent_bias = nn.Parameter(torch.Tensor(1, encode_dim))
        self.context_weight = nn.Parameter(torch.Tensor(encode_dim, 1))
        self.use_att = True
        self.init_weights(mean=0.0, std=0.05)

    def init_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.sent_bias.data.normal_(mean, std)

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.to(self.device)
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            if node[i][0] != -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] != -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        batch_current = batch_current.index_copy(0, Variable(self.th.LongTensor(index).to(self.device)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node).to(self.device)))) # do not need to transform if use GRU

        childs_hidden_sum = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))

        hidden_per_child = []   # for calculating attention
        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                cur_child_hidden = zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c]).to(self.device)), tree)
                childs_hidden_sum += cur_child_hidden
                hidden_per_child.append(cur_child_hidden)
        # batch_current = F.tanh(batch_current)

        # attention computation
        if self.use_att and (len(hidden_per_child) != 0):
            child_hiddens = torch.stack(hidden_per_child)
            childs_weighted = matrix_mul(child_hiddens, self.sent_weight, self.sent_bias)
            childs_weighted = matrix_mul(childs_weighted, self.context_weight).permute(1, 0)
            childs_weighted = F.softmax(childs_weighted)
            childs_hidden_sum = element_wise_mul(child_hiddens, childs_weighted.permute(1, 0)).squeeze(0)

        batch_current = batch_current.unsqueeze(0)
        childs_hidden_sum = childs_hidden_sum.unsqueeze(0)
        childs_weighted, hn = self.agg_net(batch_current, childs_hidden_sum)
        hn = hn.squeeze(0)

        batch_index = [i for i in batch_index if i != -1]
        b_in = Variable(self.th.LongTensor(batch_index).to(self.device))
        nd_tmp = self.batch_node.index_copy(0, b_in, hn)
        self.node_list.append(nd_tmp)

        return hn

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        h_root = self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]
        # return h_root


class BatchProgramClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim,
                 label_size, batch_size, device, use_gpu=True, pretrained_weight=None):
        super(BatchProgramClassifier, self).__init__()
        self.stop = [vocab_size-1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        self.device = device
        #class "BatchTreeEncoder"
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, self.device, pretrained_weight)
        self.root2label = nn.Linear(self.encode_dim, self.label_size)
        # gru
        self.bilstm = nn.LSTM(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.encode_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        # linear
        self.gruout2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        self.transformerout2label = nn.Linear(self.encode_dim, self.label_size)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bilstm, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(self.device))
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(self.device))
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).to(self.device))
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.to(self.device)
        return zeros

    def forward(self, x):
        filter_tree = []
        for tree in x:
            filter_subtree = []
            for sub_tree in tree:
                if len(sub_tree) > 1:
                    filter_subtree.append(sub_tree)
            filter_tree.append(filter_subtree)
        x = filter_tree

        lens = [len(item) for item in x]
        max_len = max(lens)
        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)  # 32 * 105 * 128

        # transformer
        out = self.transformer_encoder(encodes)
        out = torch.transpose(out, 1, 2)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        y = self.transformerout2label(out)
        return y

