from typing import Dict

ROLES = ["Retailer", "Wholesaler", "Distributor", "Manufacturer", "MultiFacility"]


def get_state_name_to_index_mapping(state_version: str) -> Dict[str, int]:
    if state_version == "v0":
        mapping = {
            "on_hand": 0,
            "unfilled_demand": 1,
            "latest_demand": 2,
            "unreceived_pipeline": [3, 4, 5, 6],
        }
    elif state_version == "v1":
        mapping = {
            "on_hand": 0,
            "unfilled_demand": 1,
            "latest_demand": 2,
            "on_order": 3,
            "unreceived_pipeline": [4, 5, 6, 7],
        }
    else:
        raise ValueError()

    return mapping


def get_state_len(mapping: Dict) -> int:
    """
    count the dimension of the state
    Args:
        mapping:

    Returns:

    """
    cnt = 0
    for key, value in mapping.items():
        if isinstance(value, int):
            cnt += 1
        elif isinstance(value, list):
            cnt += len(value)
        else:
            raise TypeError()

    return cnt
