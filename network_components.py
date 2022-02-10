from dataclasses import dataclass
import warnings
from typing import Union, List, Optional
from utils.demands import Demand
import numpy as np


class Order:
    """A (production or purchase) order object that is initiated by a node_name.

    Attributes:
        order_quantity (float or int):
        shipped_quantity (float or int):
        remaining_lead_time (int): remaining lead time till the supplier observes this order.

    """
    def __init__(self, order_quantity, shipped_quantity: float = 0, remaining_lead_time: int = 0,
                 integer_order=False) -> None:
        if isinstance(order_quantity, (float, int, np.generic)):
            self.order_quantity = order_quantity
        elif isinstance(order_quantity, Demand):
            self.order_quantity = order_quantity.generator()
        else:
            raise TypeError(
                f'order_quantity must be an instance of float, int or DemandGenerator, got {type(order_quantity)}'
                f'{order_quantity}')

        if integer_order:
            shipped_quantity = int(shipped_quantity)

        self.shipped_quantity = shipped_quantity
        self.remaining_lead_time = remaining_lead_time

    def __str__(self):
        return 'Order(order_quantity: {}, shipped_quantity: {}, remaining_lead_time: {})'.format(
            self.order_quantity, self.shipped_quantity, self.remaining_lead_time)

    def __repr__(self):
        return self.__str__()

    @property
    def unshipped_quantity(self):
        return self.order_quantity - self.shipped_quantity

    @property
    def requires_shipping(self):
        # requires a shipment if order is received by the suppler and unshipped quantity is positive
        return (self.remaining_lead_time <= 0) and (self.shipped_quantity < self.order_quantity)


@dataclass
class Shipment:
    quantity: float
    time_till_arrival: int


class OrderList(list):

    @property
    def requires_shipment_subtotal(self):
        return sum([so.unshipped_quantity for so in self if so.remaining_lead_time <= 0])

    def clean_finished_orders(self):
        self.sort(key=lambda x: x.unshipped_quantity, reverse=True)
        while (self.__len__() > 0) and (self[-1].unshipped_quantity == 0):
            self.pop()


class ShipmentList(list):

    def receive_shipments(self):
        arrived_quantity = 0
        self.sort(key=lambda x: x.time_till_arrival, reverse=True)
        while (self.__len__() > 0) and (self[-1].time_till_arrival <= 0):
            popped_shipment = self.pop()
            arrived_quantity += popped_shipment.quantity

        return arrived_quantity

    @property
    def en_route_subtotal(self):
        return sum([sm.quantity for sm in self])


class Arc:
    """ An Arc object the defines the relationship between two nodes in the supply network

    Attributes:
        source (str): name of the supplier node_name
        target (str): name of the customer node_name
        sales_orders (OrderList): A list of sales orders

        ordering_cost (float): A fixed cost that incurs everytime an order is placed regardless of the order quantity. (TODO)


    """
    HISTORY_LEN = 4  # TODO

    def __init__(self, source: str, target: str, information_leadtime, shipment_leadtime,
                 initial_shipments: Optional[List] = None, initial_sales_orders: Optional[List] = None,
                 initial_previous_orders=None,
                 random_init=False,
                 ordering_cost: float = .0):
        self.source = source
        self.target = target
        self.information_leadtime = information_leadtime
        self.shipment_leadtime = shipment_leadtime

        self.initial_shipments = initial_shipments
        self.initial_SOs = initial_sales_orders

        self.initial_previous_orders = initial_previous_orders
        self.random_init = random_init

        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        if self.initial_shipments is None:
            shmts = [0] * self.shipment_leadtime
        elif self.random_init:
            shmts = [np.random.randint(self.initial_shipments[0], self.initial_shipments[1]) for t in
                     range(self.shipment_leadtime)]
        else:
            shmts = self.initial_shipments[:self.shipment_leadtime]

        self.shipments = ShipmentList(
            [Shipment(shmts[t], t + 1) for t in range(self.shipment_leadtime)])

        if self.initial_SOs is None:
            sos = [0] * self.information_leadtime
        elif self.random_init:
            sos = [np.random.randint(self.initial_SOs[0], self.initial_SOs[1]) for t in
                   range(self.information_leadtime)]
        else:
            sos = self.initial_SOs[:self.information_leadtime]

        self.sales_orders = OrderList(
            [Order(sos[t], 0, t + 1) for t in range(self.information_leadtime)])

        self.previous_orders = ([0] * 4 + shmts + sos)[::-1][:self.HISTORY_LEN]

        self.unreceived_quantities = ([0] * 4 + shmts + sos)[::-1][:self.HISTORY_LEN+1]

    def keep_order_history(self, order_quantity):
        """Track order history for reporting state

        """
        if len(self.previous_orders) >= self.HISTORY_LEN:
            self.previous_orders.pop()
        self.previous_orders.insert(0, order_quantity)

    def advance_order_slips(self):
        for so in self.sales_orders:
            so.remaining_lead_time -= 1

    def update_latest_demand(self):
        latest_demand = 0
        for so in self.sales_orders:
            if so.remaining_lead_time == 0:
                latest_demand += so.order_quantity

        return latest_demand

    def advance_and_receive_shipments(self):

        # advance shipments
        for shipment in self.shipments:
            shipment.time_till_arrival -= 1

        # receive shipments
        arrived_quantity = self.shipments.receive_shipments()
        return arrived_quantity

    def receive_shipments(self):
        arrived_quantity = self.shipments.receive_shipments()
        return arrived_quantity

    def fill_orders(self, node) -> float:
        """Fulfills (ships) outstanding sales orders.

        Args:
            node (Node):

        Returns:
            Outstanding quantity that is not fulfilled due to insufficient inventory

        """
        unfilled_quantity = 0

        for so in self.sales_orders:
            if so.requires_shipping:

                # quantity of the new shipment should be minimum between available inventory and unshipped quantity
                quantity = min(node.current_inventory, so.unshipped_quantity)
                if quantity > 0:
                    self.shipments.append(Shipment(quantity, self.shipment_leadtime))
                    so.shipped_quantity += quantity

                if not node.is_external_supplier:
                    node.current_inventory -= quantity

                unfilled_quantity += so.unshipped_quantity

        # clean up finished orders
        self.sales_orders.clean_finished_orders()

        return unfilled_quantity

    def __str__(self):
        return 'arc(source:{}, target:{}, information leadtime:{}, shipment leadtime:{})'.format(
            self.source, self.target, self.information_leadtime, self.shipment_leadtime)

    def __repr__(self):
        return self.__str__()


class Node:
    """A node_name object that represents a facility in the supply chain, for example a retailer or a wholesaler

    Attributes:
        name (str): name of the node_name
        policy : this node_name's predefined ordering policy
        is_demand_source (bool): whether this node_name has external (independent) demand
        is_external_supplier (bool): if `True`, then this node_name has unlimited capacity and does not incur cost, but may
            still be subject to leadtimes.

    """

    def __init__(self, name, policy=None, is_demand_source: bool = False, demands=None,
                 is_external_supplier: bool = False,
                 initial_inventory: Union[int, float, list] = 0.0, holding_cost: float = 0.5,
                 backorder_cost: float = 1.0, setup_cost: float = .0):

        self.name = name
        self.demands = demands
        self.policy = policy

        self.is_demand_source = is_demand_source  # 'demand node_name' with independent demand
        self.current_external_demand = None
        self.is_external_supplier = is_external_supplier

        self.unit_holding_cost = holding_cost
        self.unit_backorder_cost = backorder_cost
        self.setup_cost = setup_cost

        self.initial_inventory = 9999999.0 if is_external_supplier else initial_inventory

        self.reset()

    def __str__(self):
        return f'Node({self.name}: Inventory: {self.current_inventory}, Unfilled Demand: {self.unfilled_demand}'

    def __repr__(self):
        return self.__str__()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        if type(self.initial_inventory) in [int, float]:
            self.current_inventory = self.initial_inventory
        elif type(self.initial_inventory) is list:
            self.current_inventory = np.random.randint(self.initial_inventory[0], self.initial_inventory[1])
        else:
            raise TypeError(f'type {type(self.initial_inventory)} not supported')

        self.unfilled_demand = 0
        self.unfilled_independent_demand = 0

        self.current_backorder_cost = 0
        self.current_holding_cost = 0

        self.latest_demand = []

        self.backorder_cost_history = []
        self.holding_cost_history = []
        self.order_history = []

        self.current_external_demand = 0
        if self.is_demand_source:
            self.demands.reset()
            self.demand_gen = self.demands.generator()
            self.current_external_demand = next(self.demand_gen)

    def fill_independent_demand(self) -> float:
        """Fulfills (ships) outstanding independent demand

        Returns:
            Total independent demand quantity (including ones from previous periods)
            that is not fulfilled due to insufficient inventory
        """
        if self.is_demand_source:
            quantity = max(0, min(self.current_inventory, self.current_external_demand + self.unfilled_independent_demand))
            self.current_inventory -= quantity

            unfilled_quantity = self.current_external_demand + self.unfilled_independent_demand - quantity
            self.unfilled_independent_demand = unfilled_quantity

        else:
            raise RuntimeError('A node can only fulfill independent demand when is_demand_source is True')

        return self.unfilled_independent_demand

    def update_demand(self):
        self.current_external_demand = next(self.demand_gen)

    def place_order(self, obs: Union[dict, np.ndarray], arc: Arc, order_quantity=None):
        """Place an order for the specified arc, and update the list of sales orders
        """

        if order_quantity is None:
            # No quantity is provided, fallback to the node policy
            if isinstance(obs, np.ndarray):
                order_quantity = self.policy.get_order_quantity(obs).item()
            else:
                order_quantity = self.policy.get_order_quantity({self.name: obs}).item()

        if order_quantity < 0:
            warnings.warn(f'order quantity is {order_quantity} but it should be non-negative. '
                          f'Quantity will be truncated to 0', category=RuntimeWarning)
            order_quantity = 0

        arc.keep_order_history(order_quantity)

        new_order = Order(order_quantity, 0, arc.information_leadtime)

        arc.sales_orders.append(new_order)

        arc.unreceived_quantities.insert(0, order_quantity)
        self.order_history.append(order_quantity)
