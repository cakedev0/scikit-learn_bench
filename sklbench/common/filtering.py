from typing import List

from ..config import BenchCase
from ..utils.common import hash_from_json_repr


def bench_case_filter(bench_case: BenchCase, filters: List[BenchCase]):
    if not filters:
        return True
    original_hash = hash_from_json_repr(bench_case.json_dict())
    filtered_hashes = [
        hash_from_json_repr(bench_filter.json_dict())
        for bench_filter in filters
    ]
    return original_hash in filtered_hashes
