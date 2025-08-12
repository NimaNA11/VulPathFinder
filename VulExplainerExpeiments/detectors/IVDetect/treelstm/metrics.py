from copy import deepcopy
import torch


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        

    def pearson(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        return torch.mean((x - y) ** 2)

    def calculate_evaluation_orders(self, tree):
        """
        Calculate the evaluation order of nodes in a tree.

        Args:
            tree: A tree structure where each node has a list of children.

        Returns:
            A list of nodes in post-order traversal.
        """
        evaluation_order = []

        def post_order_traversal(node):
            for child in node.children:
                post_order_traversal(child)
            evaluation_order.append(node)

        # Start traversal from the tree itself, as it represents the root node
        post_order_traversal(tree)
        return evaluation_order