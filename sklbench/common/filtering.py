from copy import deepcopy
from typing import Dict, List

from ..utils.common import hash_from_json_repr
from ..utils.custom_types import BenchCase


def merge_dicts(first: Dict, second: Dict) -> Dict:
    result = deepcopy(first)
    for key, value in second.items():
        if key not in result:
            result[key] = deepcopy(value)
        else:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, dict):
                result[key] = [merge_dicts(el, deepcopy(value)) for el in result[key]]
            elif isinstance(result[key], dict) and isinstance(value, list):
                result[key] = [merge_dicts(result[key], deepcopy(el)) for el in value]
            elif isinstance(result[key], list) and isinstance(value, list):
                local_result = []
                for element_in_first in result[key]:
                    for element_in_second in value:
                        local_result.append(
                            merge_dicts(element_in_first, element_in_second)
                        )
                result[key] = local_result
            else:
                result[key] = deepcopy(value)
    return result


def bench_case_filter(bench_case: BenchCase, filters: List[BenchCase]):
    original_hash = hash_from_json_repr(bench_case)
    filtered_hashes = [
        hash_from_json_repr(merge_dicts(bench_case, bench_filter))
        for bench_filter in filters
    ]
    return original_hash in filtered_hashes or len(filtered_hashes) == 0
