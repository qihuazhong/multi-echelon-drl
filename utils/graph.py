from typing import List
import yaml


def parse_order_sequence(nodes: List[str], customers_dict: dict) -> list:
    """Determine the sequence of placing orders. The sequence should satisfy that:
    a node_name should place order(s) only after all its customer(s) finish ordering.

    Returns:
        A list of node_name names. The first node_name orders first.
    """

    not_ordered = [node for node in nodes]
    not_ordered.sort()  # Sort the nodes by names to ensure deterministic order
    acted = {node_name: False for node_name in not_ordered}
    action_sequence = []

    def search_node(node_name: str):
        for customer in customers_dict[node_name]:
            if acted[customer]:
                pass
            else:
                search_node(customer)

        acted[node_name] = True
        not_ordered.remove(node_name)
        action_sequence.append(node_name)

    while len(not_ordered) > 0:
        search_node(not_ordered[0])

    return action_sequence


def read_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        network_config = yaml.safe_load(file)

    return network_config


# TODO
def read_json(file_path: str) -> dict:
    raise NotImplemented
