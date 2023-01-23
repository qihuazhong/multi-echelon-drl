import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict
from network_components import Node, Arc
from utils import graph
from utils.demands import Demand


class SupplyNetwork:
    """ A supply network object that contains facilities (nodes) and supplier-customer relations (arcs).

    Attributes:
        nodes: dict of Node objects, keyed by node_name name
        arcs: dict of Arc objects, key by a two-tuple of node_name names (source, target)  # TODO considering changing this
        policies (optional): dict of predefined ordering policies, keyed by node_name name

        order_sequence (List[str]): Sequence order of the information flow ()
        current_cost : cost of the current period (reward signal)
    """

    def __init__(self, nodes: List[Node], arcs: List[Arc], agent_managed_facilities: List[str], policies=None):

        self.nodes: Dict[str, Node] = {node.name: node for node in nodes}
        self.customers_dict = defaultdict(list)
        for node in self.nodes:
            self.customers_dict[node] += [arc.target for arc in arcs if arc.source == node]

        self.internal_nodes = [node.name for node in nodes if
                               not node.is_external_supplier]

        # order of the information flow
        self.order_sequence = graph.parse_order_sequence([node for node in self.nodes], self.customers_dict)
        # order of the shipment flow
        self.shipment_sequence = [self.order_sequence[i] for i in range(len(self.order_sequence) - 1, -1, -1)]

        self.agent_managed_facilities = agent_managed_facilities
        self.agent_indexes = [self.order_sequence.index(player) for player in self.agent_managed_facilities]
        # self.agent_indexes = [self.order_sequence.index(player) for player in self.internal_nodes]

        if policies:
            self.policies = policies
        else:
            self.policies = {node_name: None for node_name in self.internal_nodes}

        self.arcs: Dict[tuple, Arc] = {(arc.source, arc.target): arc for arc in arcs}
        self.demand_sources: List[str] = [node.name for node in nodes if node.is_demand_source]
        # print(f'demand sources', self.demand_sources)

        # self.supply_sources = [node_name.name for node_name in nodes if node_name.is_external_supplier]
        self.suppliers: Dict[str, list] = {node.name: [arc.source for arc in arcs if arc.target == node.name] for node in nodes}

        self.current_cost = 0

    def __str__(self):
        string = ''
        for arc in self.arcs:
            string += '{} -> {} \n'.format(self.arcs[arc].source, self.arcs[arc].target)
        return string

    def __repr__(self):
        return self.__str__()

    def summary(self):
        print('=' * 10, 'nodes:', '=' * 10, )
        for node in self.shipment_sequence:
            print('Node: {}'.format(node))
            print('\tInventory: {}'.format(self.nodes[node].current_inventory))
            print('\tUnfilled Demand: {}'.format(self.nodes[node].unfilled_demand))

        print('=' * 10, 'arcs:', '=' * 10, )
        for arc in self.arcs.keys():
            print('{} -> {}'.format(self.arcs[arc].source, self.arcs[arc].target))

            print('\tOrders')
            for so in self.arcs[arc].sales_orders:
                print('\t {}'.format(so))

            print('\tShipments')
            for shipment in self.arcs[arc].shipments:
                print('\t {}'.format(shipment))

    def reset(self, aec=False):
        for name, node in self.nodes.items():
            node.reset()

        for key, arc in self.arcs.items():
            arc.reset()

            if aec:
                last = arc.unreceived_quantities.pop()
                arc.unreceived_quantities[-1] += last

        self.current_cost = 0

        # for demand_source in self.demand_sources:
        #     for supplier in self.suppliers[demand_source]:
        #         # TODO: need to send multiple arcs together in the multi-supplier setting
        #         arc = self.arcs[(supplier, demand_source)]
        #         states = self.get_states(demand_source, 0)
        #         self.nodes[demand_source].place_order(states, arc, 0)

    def get_state(self, node_name: str) -> dict:
        """

        Returns:
            A dict object representing a node_name's state. Example:
            {suppliers: [{'name': 'supplier_1', 'last_order': 4, 'on_order': 12}, {...}],
            customers: [{'name': 'customer_1', 'latest_demand': 5, 'unfilled_demand': 2}, {...}],
            on_hand: 2
            }
        """

        M = 4  # TODO
        # On hand inventory
        on_hand = self.nodes[node_name].current_inventory

        unfilled_demand = 0

        customers = self.customers_dict[node_name]
        downstream_arcs = [self.arcs[(node_name, customer)] for customer in customers]

        unfilled_demand += sum([arc.sales_orders.requires_shipment_subtotal for arc in downstream_arcs])
        unfilled_demand += self.nodes[node_name].current_external_demand + self.nodes[node_name].unfilled_independent_demand

        # latest demand
        latest_demand = sum(self.nodes[node_name].latest_demand) + self.nodes[node_name].current_external_demand

        # on order quantity
        # purchase order quantities that have been sent out but not delivered
        suppliers = self.suppliers[node_name]
        upstream_arcs = [self.arcs[(supplier, node_name)] for supplier in suppliers]

        unshipped = sum([so.unshipped_quantity for arc in upstream_arcs for so in arc.sales_orders])
        en_route = sum([arc.shipments.en_route_subtotal for arc in upstream_arcs])

        on_order = unshipped + en_route

        states_dict = {'on_hand': on_hand,
                       'unfilled_demand': unfilled_demand,
                       'latest_demand': latest_demand,
                       # 'on_order': on_order,
                       }


        # TODO Move this to the test
        assert(-0.001 < on_order - sum([sum(arc.unreceived_quantities) for arc in upstream_arcs]) < 0.001)

        # orders_pipeline = {f'orders_pipeline_{arc.source}_{i}': arc.previous_orders[i]
        #                    for arc in upstream_arcs for i in range(len(arc.previous_orders))}

        unreceived_quantity_pipeline = {f'unreceived_pipeline_{i}': arc.unreceived_quantities[i]
                                        for arc in upstream_arcs for i in range(M)}

        states_dict = {**states_dict, **unreceived_quantity_pipeline}

        return states_dict

    def transition(self):
        """The transition that the environment performs after all the possible_agents take their actions (i.e. the
        step() function for the "environment agent" in an AEC games)

         Order of operations:
            1. Advance order slips (from customers_dict to suppliers)
            2. Advance shipments (from suppliers to customers_dict)
            3. Fulfill orders
            4. cost keeping TODO: should this be separated?

        """

        # 1. Advance order slips (from customers_dict to suppliers)
        for node in self.order_sequence:
            if self.nodes[node].is_demand_source:
                self.nodes[node].update_demand()

            for supplier in self.suppliers[node]:
                arc = self.arcs[(supplier, node)]
                arc.advance_order_slips()

                # state = self.get_states(node)
                # self.nodes[node_name].place_order(state, arc, self.time)

                latest_demand = arc.update_latest_demand()
                self.nodes[supplier].latest_demand = []  # TODO
                self.nodes[supplier].latest_demand.append(latest_demand)

                last = arc.unreceived_quantities.pop()
                arc.unreceived_quantities[-1] += last


        # 2.advance shipments & 3.Fulfill orders
        for node in self.shipment_sequence:
            for customer in self.customers_dict[node]:
                # Increase customer's inventory when the shipments arrive
                arrived_quantity = self.arcs[(node, customer)].advance_and_receive_shipments()
                self.nodes[customer].current_inventory += arrived_quantity

                consumed = 0
                for i in range(len(arc.unreceived_quantities)-1, -1, -1):
                    if arrived_quantity - consumed > 0:
                        filled_qty = min(arc.unreceived_quantities[i], arrived_quantity - consumed)
                        arc.unreceived_quantities[i] -= filled_qty
                        consumed += filled_qty

            self.nodes[node].unfilled_demand = 0
            for customer in self.customers_dict[node]:
                self.nodes[node].unfilled_demand += self.arcs[(node, customer)].fill_orders(self.nodes[node])
            if self.nodes[node].is_demand_source:
                self.nodes[node].unfilled_demand += self.nodes[node].fill_independent_demand()

        self.current_cost = self.get_cost()

        # self.time += 1

    """
    Order of operations:        
        1. Advance order slips
        2. Place orders (from customers_dict to suppliers)
        3. Advance shipments (from suppliers to customers_dict)
        4. Fulfill orders 
        5. cost keeping
    """

    def before_action(self, period):
        # place new orders & advance order slips


        for node in self.order_sequence:
            for supplier in self.suppliers[node]:
                # TODO: need to send multiple arcs together in the multi-supplier setting
                arc = self.arcs[(supplier, node)]
                arc.advance_order_slips()

                latest_demand = arc.update_latest_demand()
                # print(f'advancing order slips {node} -> {supplier}: {latest_demand}')
                self.nodes[supplier].latest_demand = []  # TODO
                # print(f'supplier: {supplier}, latest_demand:{latest_demand}')
                self.nodes[supplier].latest_demand.append(latest_demand)
                # print(f'supplier: {supplier}, Total latest_demand:{self.nodes[supplier].latest_demand}')

                last = arc.unreceived_quantities.pop()
                arc.unreceived_quantities[-1] += last

        for node in self.order_sequence[:min(self.agent_indexes)]:

            for supplier in self.suppliers[node]:
                # TODO: need to send multiple arcs together in the multi-supplier setting

                arc = self.arcs[(supplier, node)]
                # arc.advance_order_slips()


                states = self.get_state(node)
                self.nodes[node].place_order(states, arc)



    def observations(self, agent: str):
        return self.get_state(agent)

    def agent_action(self, period, order_quantities):
        action_dim_counter = 0
        for node in self.agent_managed_facilities:

            for supplier in self.suppliers[node]:
                # TODO: need to send multiple arcs together in the multi-supplier setting
                arc = self.arcs[(supplier, node)]
                # arc.advance_order_slips()

                states = self.get_state(node)
                self.nodes[node].place_order(states, arc, order_quantity=order_quantities[action_dim_counter])


            action_dim_counter += 1

    def after_action(self, period):
        # place new orders
        for node in self.order_sequence[max(self.agent_indexes) + 1:]:
            for supplier in self.suppliers[node]:
                # TODO: need to send multiple arcs together in the multi-supplier setting
                arc = self.arcs[(supplier, node)]
                # arc.advance_order_slips()

                states = self.get_state(node)
                self.nodes[node].place_order(states, arc)


        # 3.advance shipments & 4.Fulfill orders
        for node in self.shipment_sequence:
            for customer in self.customers_dict[node]:
                # Increase customer's inventory when the shipments arrive
                arc = self.arcs[(node, customer)]
                arrived_quantity = arc.advance_and_receive_shipments()
                self.nodes[customer].current_inventory += arrived_quantity

                # print(f'arrived_quantity {arrived_quantity}')
                # print(f'before consumption {arc.unreceived_quantities}')
                consumed = 0
                for i in range(len(arc.unreceived_quantities)-1, -1, -1):
                    if arrived_quantity - consumed > 0:
                        filled_qty = min(arc.unreceived_quantities[i], arrived_quantity - consumed)
                        arc.unreceived_quantities[i] -= filled_qty
                        consumed += filled_qty
                # print(f'after consumption {arc.unreceived_quantities}')

            self.nodes[node].update_last_backlog()

            self.nodes[node].unfilled_demand = 0
            for customer in self.customers_dict[node]:
                self.nodes[node].unfilled_demand += self.arcs[(node, customer)].fill_orders(self.nodes[node])
            if self.nodes[node].is_demand_source:
                self.nodes[node].unfilled_demand += self.nodes[node].fill_independent_demand()


        for node_name, node in self.nodes.items():
            if node.is_demand_source:
                node.update_demand()


    def get_node_cost(self, node_name: str) -> float:
        """
        TODO: include setup costs
        """

        current_node: Node = self.nodes[node_name]

        # inventory holding cost
        c_h = -current_node.current_inventory * current_node.unit_holding_cost

        # backorder cost
        c_b = -current_node.unfilled_demand * current_node.unit_backorder_cost

        return c_h + c_b

    def get_cost(self) -> float:
        """
        Returns:
            The cost (usually a negative real number) of the current period, used as the reward signal

        TODO: include setup costs
        """
        c_h = 0  # inventory holding cost
        c_b = 0  # backorder cost

        internal_nodes = [node for node_name, node in self.nodes.items() if
                          not node.is_external_supplier]

        for node in internal_nodes:
            # holding cost
            current_holding_cost = -node.current_inventory * node.unit_holding_cost
            # backorder cost
            current_backorder_cost = -node.unfilled_demand * node.unit_backorder_cost

            # keep_cost_history
            node.holding_cost_history.append(current_holding_cost)
            node.backorder_cost_history.append(current_backorder_cost)

            c_h += current_holding_cost
            c_b += current_backorder_cost

        return c_h + c_b

    # TODO: move this function to an analysis module/script
    def get_cost_history(self, nodes=None, as_df=False):
        """
        Get cost history of a node_name.

        Args:
            nodes: list or String. Names of the cost nodes. If not provided, will return cost history of all nodes.
            as_df: Return cost history as pandas DataFrame object.
        """

        if isinstance(nodes, str):
            nodes = [nodes]
        elif nodes is None:
            nodes = self.internal_nodes

        cost_dict = {node: {'holding_cost': self.nodes[node].holding_cost_history,
                            'backorder_cost': self.nodes[node].backorder_cost_history} for node in nodes}

        if as_df:
            history_len = len(self.nodes[nodes[0]].holding_cost_history)
            modified_dict = {key: {'node_name': key, 'period': np.arange(history_len), **item}
                             for key, item in cost_dict.items()}
            return pd.concat([pd.DataFrame().from_dict(item) for key, item in modified_dict.items()], ignore_index=True)
        else:
            return cost_dict


def from_dict(network_config: dict) -> SupplyNetwork:
    """
    Create a SupplyNetwork object through a dictionary, which can be imported through reading yaml, json files.
    Args:
        network_config:

    Returns:

    """
    nodes: List[Node] = []
    agent_managed_facilities: List[str] = []

    # Create node_name instances according to the dictionary. Optional values are defaulted to 0 or False if not provided.
    node: dict
    for node in network_config['nodes']:
        if node.get('demand_path', None):
            is_demand_source = False
            demand_generator = None
        else:
            is_demand_source = True
            demand_generator = Demand('samples', data_path=node['demand_path'])

        if node.get('agent_managed', False):
            agent_managed_facilities.append(node['name'])

        nodes.append(Node(name=node['name'],
                          is_demand_source=is_demand_source,
                          demands=demand_generator,
                          is_external_supplier=node.get('is_external_supplier', False),
                          initial_inventory=node.get('initial_inventory', 0),
                          holding_cost=node.get('holding_cost', 0),
                          backorder_cost=node.get('backorder_cost', 0),
                          setup_cost=node.get('setup_cost', 0)))

    arcs: List[Arc] = []
    # Create arc instances. Optional values are defaulted to 0 or False if not provided.
    for arc in network_config['arcs']:
        arcs.append(Arc(source=arc['supplier'],
                        target=arc['customer'],
                        information_leadtime=arc.get('info_leadtime', 0),
                        shipment_leadtime=arc.get('shipment_leadtime', 0),
                        ordering_cost=arc.get('ordering_cost', 0)))

    sn = SupplyNetwork(nodes=nodes, arcs=arcs, agent_managed_facilities=agent_managed_facilities)
    return sn
