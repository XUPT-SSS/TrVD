import argparse
import copy
import pandas as pd
import os

from tree_sitter import Language, Parser
from prepare_data import get_root_paths

def parse_options():
    parser = argparse.ArgumentParser(description='TrVD preprocess~.')
    parser.add_argument('-i', '--input', default='mutrvd', choices='mutrvd',
                        help='training dataset type', type=str, required=False)
    args = parser.parse_args()
    return args


def parse_ast(source):
    # C_LANGUAGE = Language('build_languages/my-languages.so', 'c')
    CPP_LANGUAGE = Language('build_languages/my-languages.so', 'cpp')
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)  # set the parser for certain language
    tree = parser.parse(source.encode('utf-8').decode('unicode_escape').encode())
    return tree

args = parse_options()

class Pipeline:
    def __init__(self):
        self.train = None
        self.train_keep = None
        self.train_block = None
        self.dev = None
        self.dev_keep = None
        self.dev_block = None
        self.test = None
        self.test_keep = None
        self.test_block = None
        self.size = None
        self.w2v_path = None

    # parse source code
    def parse_source(self, dataset):
        train = pd.read_pickle('dataset/'+dataset+'/train.pkl')
        test = pd.read_pickle('dataset/'+dataset+'/test.pkl')
        dev = pd.read_pickle('dataset/'+dataset+'/val.pkl')

        # parsing source source into ast
        train['code'] = train['code'].apply(parse_ast)
        self.train = train
        self.train_keep = copy.deepcopy(train)
        dev['code'] = dev['code'].apply(parse_ast)
        self.dev = dev
        self.dev_keep = copy.deepcopy(dev)
        test['code'] = test['code'].apply(parse_ast)
        self.test = test
        self.test_keep = copy.deepcopy(test)


    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, size):
        self.size = size
        trees = self.train
        self.w2v_path = 'subtrees/'+args.input+'/node_w2v_'+str(size)
        if not os.path.exists('subtrees/'+args.input):
            os.mkdir('subtrees/'+args.input)
        from prepare_data import get_sequences
        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            # collect all root-leaf paths
            paths = []
            get_root_paths(ast, paths, [])
            # add root to leaf path as corpus
            paths.append(sequence)
            return paths
        # train word2vec embedding if not exists
        if not os.path.exists(self.w2v_path):
            corpus = trees['code'].apply(trans_to_sequences)
            paths = []
            for all_paths in corpus:
                for path in all_paths:
                    path = [token.decode('utf-8') if type(token) is bytes else token for token in path]
                    paths.append(path)
            corpus = paths
            # training word2vec model
            from gensim.models.word2vec import Word2Vec
            print('corpus size: ', len(corpus))
            w2v = Word2Vec(corpus, size=size, workers=96, sg=1, min_count=3)
            print('word2vec : ', w2v)
            w2v.save(self.w2v_path)

    def generate_block_seqs_time(self, data):
        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec
        word2vec = Word2Vec.load('subtrees/trvd/node_w2v_128').wv
        vocab = word2vec.vocab
        max_token = word2vec.vectors.shape[0]

        def tree_to_index(node):
            token = node.token
            if type(token) is bytes:
                token = token.decode('utf-8')
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def tree_to_token(node):
            token = node.token
            if type(token) is bytes:
                token = token.decode('utf-8')
            result = [token]
            children = node.children
            for child in children:
                result.append(tree_to_token(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                token_tree = tree_to_token(b)
                print(token_tree)
                tree.append(btree)
            return tree

        return trans2seq(data)

    # generate block sequences with index representations
    def generate_block_seqs(self, data, name: str):
        blocks_path = None
        if name == 'train':
            blocks_path = 'subtrees/'+args.input+'/train_block.pkl'
        elif name == 'test':
            blocks_path = 'subtrees/'+args.input+'/test_block.pkl'
        elif name == 'dev':
            blocks_path = 'subtrees/'+args.input+'/dev_block.pkl'

        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec
        word2vec = Word2Vec.load(self.w2v_path).wv
        vocab = word2vec.vocab
        max_token = word2vec.vectors.shape[0]

        def tree_to_index(node):
            token = node.token
            if type(token) is bytes:
                token = token.decode('utf-8')
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def tree_to_token(node):
            token = node.token
            if type(token) is bytes:
                token = token.decode('utf-8')
            result = [token]
            children = node.children
            for child in children:
                result.append(tree_to_token(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                token_tree = tree_to_token(b)
                print(token_tree)
                tree.append(btree)
            return tree

        trees = data
        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_pickle(blocks_path)
        return trees


    # run for processing raw to train
    def run(self, dataset):
        print('parse source code...')
        self.parse_source(dataset)
        print('train word2vec model...')
        self.dictionary_and_embedding(size=128)
        print('generate block sequences...')
        self.train_block = self.generate_block_seqs(self.train_keep, 'train')
        self.dev_block = self.generate_block_seqs(self.dev_keep, 'dev')
        self.test_block = self.generate_block_seqs(self.test_keep, 'test')


if __name__ == '__main__':
    ppl = Pipeline()
    print('Now precessing dataset: ', args.input)
    ppl.run(args.input)


