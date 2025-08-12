from . import Constants
from .dataset import SICKDataset
from .metrics import Metrics
from .model import SimilarityTreeLSTM
from .trainer import Trainer
from .tree import Tree
from . import utils
from .vocab import Vocab

metrics = Metrics(num_classes=2)
# calculate_evaluation_orders = metrics.calculate_evaluation_orders(Tree)
def calculate_evaluation_orders(edges, num_nodes):
    """
    Calculate the evaluation order of nodes and edges in a tree.

    Args:
        edges: A list of tuples representing edges in the tree (parent, child).
        num_nodes: The total number of nodes in the tree.

    Returns:
        node_order: A list of nodes in post-order traversal.
        edge_order: A list of edges in the order they are evaluated.
    """
    from collections import defaultdict, deque

    # Build adjacency list and in-degree count
    children = defaultdict(list)
    in_degree = [0] * num_nodes
    for parent, child in edges:
        children[parent].append(child)
        in_degree[child] += 1

    # Find the root node (node with in-degree 0)
    root = next(i for i in range(num_nodes) if in_degree[i] == 0)

    # Perform post-order traversal
    node_order = []
    edge_order = []

    def post_order_traversal(node):
        for child in children[node]:
            post_order_traversal(child)
            edge_order.append((node, child))
        node_order.append(node)

    post_order_traversal(root)
    return node_order, edge_order

__all__ = [Constants, SICKDataset, Metrics, SimilarityTreeLSTM, Trainer, Tree, Vocab, utils]#, calculate_evaluation_orders]
