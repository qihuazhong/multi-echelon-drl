from functools import lru_cache
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from network_components import Node, Arc
from utils import graph
from utils.demands import Demand
from utils.heuristics import BaseStockPolicy


class SupplyNetwork:
    """A supply network object that contains facilities (nodes) and supplier-customer relations (arcs).

    Attributes:
        nodes: dict of Node objects, keyed by node_name name
        arcs: dict of Arc objects, key by a two-tuple of node_name names (source, target)  # TODO considering changing this
        policies (optional): dict of predefined ordering policies, keyed by node_name name

        order_sequence (List[str]): Sequence order of the information flow ()
        current_cost : cost of the current period (reward signal)
    """

    def __init__(
        self,
        nodes: List[Node],
        arcs: List[Arc],
        agent_managed_facilities: List[str],
        policies=None,
        cost_type: str = "general",
        state_version: str = "v0",
    ):
        self.nodes: Dict[str, Node] = {node.name: node for node in nodes}
        self.customers_dict = defaultdict(list)
        for node in self.nodes:
            self.customers_dict[node] += [
                arc.target for arc in arcs if arc.source == node
            ]

        self.internal_nodes = [
            node.name for node in nodes if not node.is_external_supplier
        ]

        # order of the information flow
        self.order_sequence: List[str] = graph.parse_order_sequence(
            [node for node in self.nodes], self.customers_dict
        )
        # order of the shipment flow
        self.shipment_sequence: List[str] = [
            self.order_sequence[i] for i in range(len(self.order_sequence) - 1, -1, -1)
        ]

        self.agent_managed_facilities = agent_managed_facilities
        self.agent_indexes = [
            self.order_sequence.index(player)
            for player in self.agent_managed_facilities
        ]

        if policies:
            self.policies = policies
        else:
            self.policies = {node_name: None for node_name in self.internal_nodes}

        self.arcs: Dict[Tuple[str, str], Arc] = {
            (arc.source, arc.target): arc for arc in arcs
        }
        self.demand_sources: List[str] = [
            node.name for node in nodes if node.is_demand_source
        ]
        # print(f'demand sources', self.demand_sources)

        # self.supply_sources = [node_name.name for node_name in nodes if node_name.is_external_supplier]
        self.suppliers: Dict[str, List[str]] = {
            node.name: [arc.source for arc in arcs if arc.target == node.name]
            for node in nodes
        }

        self.cost_type: str = cost_type  # "general", "clark-scarf" or "fixed-cost"
        # self.in_transit_holding_cost: bool = False  # whether the outgoing shipment accrue holding cost until received.

        self.state_version = state_version
        self.current_cost = 0

    def __str__(self):
        string = ""
        for arc in self.arcs:
            string += "{} -> {} \n".format(self.arcs[arc].source, self.arcs[arc].target)
        return string

    def __repr__(self):
        return self.__str__()

    @lru_cache
    def get_customer_names(self, node_name: str) -> List[str]:
        """Given a node name, return the list of the node's customer names"""

        arc_keys: List[Tuple[str]] = list(self.arcs.keys())

        return [target for source, target in arc_keys if source == node_name]

    @lru_cache
    def get_outgoing_arcs(self, node_name: str) -> List[Arc]:
        return [
            arc for (source, target), arc in self.arcs.items() if source == node_name
        ]

    @lru_cache
    def get_incoming_arcs(self, node_name: str) -> List[Arc]:
        return [
            arc for (source, target), arc in self.arcs.items() if target == node_name
        ]

    def summary(self):
        print(
            "=" * 10,
            "nodes:",
            "=" * 10,
        )
        for node in self.shipment_sequence:
            print("Node: {}".format(node))
            print("\tInventory: {}".format(self.nodes[node].current_inventory))
            print("\tUnfilled Demand: {}".format(self.nodes[node].unfilled_demand))

        print(
            "=" * 10,
            "arcs:",
            "=" * 10,
        )
        for arc in self.arcs.keys():
            print("{} -> {}".format(self.arcs[arc].source, self.arcs[arc].target))

            print("\tOrders")
            for so in self.arcs[arc].sales_orders:
                print("\t {}".format(so))

            print("\tShipments")
            for shipment in self.arcs[arc].shipments:
                print("\t {}".format(shipment))

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

        unfilled_demand += sum(
            [arc.sales_orders.requires_shipment_subtotal for arc in downstream_arcs]
        )
        unfilled_demand += (
            self.nodes[node_name].current_external_demand
            + self.nodes[node_name].unfilled_independent_demand
        )

        # latest demand
        latest_demand = (
            sum(self.nodes[node_name].latest_demand)
            + self.nodes[node_name].current_external_demand
        )

        # on order quantity
        # purchase order quantities that have been sent out but not delivered
        suppliers = self.suppliers[node_name]
        upstream_arcs = [self.arcs[(supplier, node_name)] for supplier in suppliers]

        unshipped = sum(
            [so.unshipped_quantity for arc in upstream_arcs for so in arc.sales_orders]
        )
        en_route = sum([arc.shipments.en_route_subtotal for arc in upstream_arcs])

        on_order = unshipped + en_route

        states_dict = {
            "on_hand": on_hand,
            "unfilled_demand": unfilled_demand,
            "latest_demand": latest_demand,
            # "on_order": on_order,
        }

        # TODO Move this to the test
        assert (
            -0.001
            < on_order - sum([sum(arc.unreceived_quantities) for arc in upstream_arcs])
            < 0.001
        )

        # orders_pipeline = {f'orders_pipeline_{arc.source}_{i}': arc.previous_orders[i]
        #                    for arc in upstream_arcs for i in range(len(arc.previous_orders))}

        unreceived_quantity_pipeline = {
            f"unreceived_pipeline_{i}": arc.unreceived_quantities[i]
            for arc in upstream_arcs
            for i in range(M)
        }

        if self.state_version == "v0":
            pass
        elif self.state_version == "v1":
            states_dict["on_order"] = on_order

            arc = upstream_arcs[0]
            # unreceived_quantity_pipeline[f"unreceived_pipeline_{3}"] = sum(arc.unreceived_quantities[i] for arc in upstream_arcs for i in range(M))
            # unreceived_quantity_pipeline[f"unreceived_pipeline_{2}"] = arc.sales_orders.unshipped_subtotal

            unreceived_quantity_pipeline[
                f"unreceived_pipeline_{3}"
            ] = arc.previous_orders[0]
            unreceived_quantity_pipeline[
                f"unreceived_pipeline_{2}"
            ] = arc.previous_orders[1]

            pipeline_len = 2
            if arc.shipment_leadtime >= 3:
                pipeline_len = 4

            for i in range(pipeline_len):
                unreceived_quantity_pipeline[
                    f"unreceived_pipeline_{i}"
                ] = arc.shipments.shipment_quantity_by_time[i]

        else:
            raise ValueError

        states_dict = {**states_dict, **unreceived_quantity_pipeline}

        return states_dict

    """
    Order of operations:        
        1. Advance order slips
        2. Place orders (from customers_dict to suppliers)
        3. Advance shipments (from suppliers to customers_dict)
        4. Fulfill orders 
        5. cost keeping
    """

    def pre_agent_action(self):
        # place new orders & advance order slips

        for node in self.order_sequence:
            for supplier in self.suppliers[node]:
                # TODO: need to send multiple arcs together in the multi-supplier setting
                arc = self.arcs[(supplier, node)]
                arc.advance_order_slips()

                # latest_demand = arc.update_latest_demand()
                # self.nodes[supplier].latest_demand = []  # TODO
                # self.nodes[supplier].latest_demand.append(latest_demand)

                last = arc.unreceived_quantities.pop()
                arc.unreceived_quantities[-1] += last

                if node in self.order_sequence[: min(self.agent_indexes)]:
                    states = self.get_state(node_name=node)
                    self.nodes[node].place_order(obs=states, arc=arc)

                latest_demand = arc.update_latest_demand()
                self.nodes[supplier].latest_demand = []  # TODO
                self.nodes[supplier].latest_demand.append(latest_demand)

        # for node in self.order_sequence[: min(self.agent_indexes)]:
        #
        #     for supplier in self.suppliers[node]:
        #         # TODO: need to send multiple arcs together in the multi-supplier setting
        #
        #         arc = self.arcs[(supplier, node)]
        #         # arc.advance_order_slips()
        #
        #         states = self.get_state(node_name=node)
        #         self.nodes[node].place_order(obs=states, arc=arc)

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
                self.nodes[node].place_order(
                    states, arc, order_quantity=order_quantities[action_dim_counter]
                )

            action_dim_counter += 1

    def post_agent_action(self):
        # place new orders
        for node in self.order_sequence[max(self.agent_indexes) + 1 :]:
            for supplier in self.suppliers[node]:
                # TODO: need to send multiple arcs together in the multi-supplier setting
                arc = self.arcs[(supplier, node)]
                # arc.advance_order_slips()

                states = self.get_state(node)
                self.nodes[node].place_order(obs=states, arc=arc)

        # 3.advance shipments & 4.Fulfill orders
        for node_name in self.shipment_sequence:
            for customer in self.customers_dict[node_name]:
                # Increase customer's inventory when the shipments arrive
                arc = self.arcs[(node_name, customer)]

                self.nodes[customer].last_received = arc.advance_and_receive_shipments()
                # arrived_quantity = arc.advance_and_receive_shipments()

                self.nodes[customer].current_inventory += self.nodes[
                    customer
                ].last_received

                consumed = 0
                for i in range(len(arc.unreceived_quantities) - 1, -1, -1):
                    if self.nodes[customer].last_received - consumed > 0:
                        filled_qty = min(
                            arc.unreceived_quantities[i],
                            self.nodes[customer].last_received - consumed,
                        )
                        arc.unreceived_quantities[i] -= filled_qty
                        consumed += filled_qty

            self.nodes[node_name].update_last_backlog()

            self.nodes[node_name].unfilled_demand = 0
            for customer in self.customers_dict[node_name]:
                filled, unfilled = self.arcs[(node_name, customer)].fill_orders(
                    self.nodes[node_name]
                )
                self.nodes[node_name].unfilled_demand += unfilled

            if self.nodes[node_name].is_demand_source:
                self.nodes[node_name].unfilled_demand += self.nodes[
                    node_name
                ].fill_independent_demand()

        if self.cost_type == "clark-scarf":
            for node in self.order_sequence:
                self.get_clark_scarf_cost_by_node_name(query_node_name=node)

        for node_name, node in self.nodes.items():
            if node.is_demand_source:
                node.update_demand()

    def get_clark_scarf_cost_by_node_name(self, query_node_name: str):
        c_h = 0  # inventory holding cost
        c_b = 0  # backlog cost

        echelon_stock = 0
        node = self.nodes[query_node_name]
        if node.is_demand_source:
            echelon_stock += max(0, node.current_inventory)
            node.echelon_stock = echelon_stock

        else:
            echelon_stock += (
                node.current_inventory
                - node.unfilled_demand
                + sum(
                    sum(arc.unreceived_quantities)
                    for arc in self.get_outgoing_arcs(node.name)
                )
            )

            echelon_stock += sum(
                [
                    self.nodes[customer].echelon_stock
                    for customer in self.customers_dict[query_node_name]
                ]
            )
            node.echelon_stock = echelon_stock

        if node.is_demand_source:
            c_b = -node.unfilled_demand * node.unit_backlog_cost
            c_h = -echelon_stock * 0.25

            node.inventory_history.append(echelon_stock)
            node.holding_cost_history.append(c_h)

        else:
            c_b = 0
            c_h = (-echelon_stock) * 0.25

            node.inventory_history.append(echelon_stock)
            node.holding_cost_history.append(c_h)

        node.backlog_history.append(node.unfilled_demand)
        node.backlog_cost_history.append(c_b)

        return c_h + c_b

    def get_general_cost(self) -> float:
        """
        Returns:
            The cost (usually a negative real number) of the current period, used as the reward signal

        TODO: include setup costs
        """
        c_h = 0  # inventory holding cost
        c_b = 0  # backlog cost

        internal_nodes: List[Node] = [
            node
            for node_name, node in self.nodes.items()
            if not node.is_external_supplier
        ]

        for node in internal_nodes:
            # holding cost
            current_holding_cost = -node.current_inventory * node.unit_holding_cost

            # backlog cost
            current_backlog_cost = -node.unfilled_demand * node.unit_backlog_cost

            # keep_history
            node.inventory_history.append(node.current_inventory)
            node.backlog_history.append(node.unfilled_demand)
            node.holding_cost_history.append(current_holding_cost)
            node.backlog_cost_history.append(current_backlog_cost)

            c_h += current_holding_cost
            c_b += current_backlog_cost

        return c_h + c_b

    def get_cost(self) -> float:
        """
        Returns:
            The cost (usually a negative real number) of the current period, used as the reward signal

        TODO: include setup costs
        """

        if self.cost_type == "general":
            return self.get_general_cost()

        elif self.cost_type == "clark-scarf":
            return sum(
                [
                    self.nodes[node_name].holding_cost_history[-1]
                    + self.nodes[node_name].backlog_cost_history[-1]
                    for node_name in self.internal_nodes
                ]
            )

        else:
            raise ValueError(f"Cost type {self.cost_type} not recognised")

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

        cost_dict = {
            node: {
                "holding_cost": self.nodes[node].holding_cost_history,
                "backlog_cost": self.nodes[node].backlog_cost_history,
            }
            for node in nodes
        }

        if as_df:
            history_len = len(self.nodes[nodes[0]].holding_cost_history)
            modified_dict = {
                key: {"node_name": key, "period": np.arange(history_len), **item}
                for key, item in cost_dict.items()
            }
            return pd.concat(
                [pd.DataFrame().from_dict(item) for key, item in modified_dict.items()],
                ignore_index=True,
            )
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
    for node in network_config["nodes"]:
        demand: dict = node.get("demand", None)
        if demand is None:
            is_demand_source = False
            demand_generator = None
        else:
            is_demand_source = True

            if demand["distribution"] == "Normal":
                demand_generator = Demand(
                    demand_pattern="normal", mean=demand["mean"], sd=demand["sd"]
                )
            elif demand["distribution"] == "Custom":
                demand_generator = Demand("samples", data_path=demand["path"])
            else:
                # TODO add other distributions
                raise ValueError(
                    f"demand distribution {demand['distribution']} not recognized"
                )

        if node.get("agent_managed", False):
            agent_managed_facilities.append(node["name"])

        target_stock_level = node.get("target_stock_level", None)
        if target_stock_level is not None:
            array_index = {
                "on_hand": 0,
                "unfilled_demand": 1,
                "latest_demand": 2,
                "unreceived_pipeline": [3, 4, 5, 6],
            }

            fallback_policy = BaseStockPolicy(
                target_levels=[target_stock_level],
                array_index=array_index,
                state_dim_per_facility=7,
                ub=np.inf,
            )
        else:
            fallback_policy = None

        nodes.append(
            Node(
                name=node["name"],
                is_demand_source=is_demand_source,
                demands=demand_generator,
                is_external_supplier=node.get("is_external_supplier", False),
                initial_inventory=node.get("initial_inventory", 0),
                holding_cost=node.get("holding_cost", 0),
                backlog_cost=node.get("backlog_cost", 0),
                setup_cost=node.get("setup_cost", 0),
                fallback_policy=fallback_policy,
            )
        )

    arcs: List[Arc] = []
    # Create arc instances. Optional values are defaulted to 0 or False if not provided.
    for arc in network_config["arcs"]:
        arcs.append(
            Arc(
                source=arc["supplier"],
                target=arc["customer"],
                information_leadtime=arc.get("info_leadtime", 0),
                shipment_leadtime=arc.get("shipment_leadtime", 0),
                ordering_cost=arc.get("ordering_cost", 0),
            )
        )

    sn = SupplyNetwork(
        nodes=nodes,
        arcs=arcs,
        agent_managed_facilities=agent_managed_facilities,
        cost_type=network_config["cost_type"],
    )
    return sn
