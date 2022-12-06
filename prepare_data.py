import copy

import tree_sitter

from tree import ASTNode, SingleNode


def get_sequences(node, sequence: list):
    current = SingleNode(node)

    if not isinstance(node, tree_sitter.Tree):
        name = node.type
    else:
        name = node.root_node.type

    if name == 'comment':
        return
    else:
        sequence.append(current.get_token())

    if not isinstance(node, tree_sitter.Tree):
        for child in node.children:
            get_sequences(child, sequence)
    else:
        for child in node.root_node.children:
            get_sequences(child, sequence)
    if current.get_token().lower() == 'compound_statement':
        sequence.append('End')


def get_root_paths(node, sequences: list, cur_path: list):
    '''
    collect all paths that originate from the root to each leaf node
    :param node:
    :param sequences:
    :param cur_path:
    :return:
    '''
    current = SingleNode(node)

    if current.is_leaf_node():

        if not isinstance(node, tree_sitter.Tree):
            name = node.type
        else:
            name = node.root_node.type

        if name == 'comment':
            return
        else:
            root_path = copy.deepcopy(cur_path)
            cur_token = current.get_token()
            root_path.append(cur_token)
            sequences.append(root_path)
            return
    else:

        if not isinstance(node, tree_sitter.Tree):
            name = node.type
        else:
            name = node.root_node.type

        if name == 'comment':
            return
        else:
            cur_path.append(current.get_token())
            if not isinstance(node, tree_sitter.Tree):
                for child in node.children:
                    par_path = copy.deepcopy(cur_path)
                    get_root_paths(child, sequences, par_path)
            else:
                # for _, child in node.root_node.children():
                for child in node.root_node.children:
                    par_path = copy.deepcopy(cur_path)
                    get_root_paths(child, sequences, par_path)


def needsSplitting(node, max_depth=8, max_size=40):
    '''
    split if the depth or size of the sub-tree exceeds certain thresholds
    :param node:
    :return:
    '''
    tr_depth = getMaxDepth(node)
    tr_size = getTreeSize(node)
    if tr_depth > max_depth or tr_size > max_size:
        return True
    return False


def getMaxDepth(node):
    if not node:
        return 0
    # leaf node touched
    if len(node.children) == 0:
        return 1

    children = node.children
    max_depth = -1
    for child in children:
        depth = getMaxDepth(child)
        max_depth = depth if (depth > max_depth) else max_depth
    return max_depth + 1


def getTreeSize(node):

    if not node:
        return 0
    # leaf node touched
    if len(node.children) == 0:
        return 1
    tr_size = 1
    children = node.children
    for child in children:
        node_num = getTreeSize(child)
        tr_size += node_num
    return tr_size


def get_blocks(node, block_seq):

    if isinstance(node, list):
        return
    elif not isinstance(node, tree_sitter.Tree):
        children = node.children
        name = node.type
    else:
        children = node.root_node.children
        name = node.root_node.type

    if name == 'comment':
        return

    if name in ['function_definition', 'if_statement', 'try_statement', 'for_statement', 'switch_statement',
                'while_statement', 'do_statement', 'catch_clause', 'case_statement']:
        # split further?
        do_split = needsSplitting(node)
        # print(do_split, 'tr_size ', getTreeSize(node), ' tr_depth ', getMaxDepth(node))

        if not do_split:
            block_seq.append(ASTNode(node, False))
            return
        else:
            block_seq.append(ASTNode(node, True))
            # find first compound_statement
            body_idx = 0
            for child in children:
                if child.type == 'compound_statement' or child.type == 'expression_statement':
                    break
                body_idx += 1

            skip = body_idx

            for i in range(skip, len(children)):
                child = children[i]
                if child.type == 'comment':
                    continue
                if child.type not in ['function_definition', 'if_statement', 'try_statement', 'for_statement',
                                      'switch_statement',
                                      'while_statement', 'do_statement']:
                    block_seq.append(ASTNode(child, needsSplitting(child)))
                get_blocks(child, block_seq)
    elif name == 'compound_statement':
        # block_seq.append(ASTNode(name))
        do_split = needsSplitting(node)

        if not isinstance(node, tree_sitter.Tree):
            for child in node.children:
                if child.type == 'comment':
                    continue

                if child.type not in ['if_statement', 'try_statement', 'for_statement', 'switch_statement',
                                      'while_statement', 'do_statement', 'catch_clause']:
                    block_seq.append(ASTNode(child, needsSplitting(child)))
                else:
                    get_blocks(child, block_seq)
        else:
            for child in node.root_node.children:
                if child.type == 'comment':
                    continue
                if child.type not in ['if_statement', 'try_statement', 'for_statement', 'switch_statement',
                                      'while_statement', 'do_statement', 'catch_clause', 'case_statement']:
                    block_seq.append(ASTNode(child, needsSplitting(child)))
                get_blocks(child, block_seq)
        # block_seq.append(ASTNode('End'))
    else:
        if isinstance(node, list):
            return
        elif not isinstance(node, tree_sitter.Tree):
            for child in node.children:
                get_blocks(child, block_seq)
        else:
            for child in node.root_node.children:
                get_blocks(child, block_seq)
