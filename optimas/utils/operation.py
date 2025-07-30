import hashlib
import json
from typing import Any, List, Tuple, Union

from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)


def hash_obj(obj: Any) -> str:
    """
    Compute a deterministic MD5 hash of an object using JSON serialization.

    Args:
        obj (Any): Object to hash. Must be JSON-serializable or convertible to string.

    Returns:
        str: Hex digest string of the object's hash.
    """
    try:
        obj_str = json.dumps(obj, sort_keys=True, separators=(",", ":"))  # compact + deterministic
    except TypeError:
        logger.warning(f"Falling back to str() for non-JSON-serializable object: {obj!r}")
        obj_str = str(obj)
    return hashlib.md5(obj_str.encode("utf-8")).hexdigest()


def is_same(obj1: Any, obj2: Any) -> bool:
    """
    Compare two objects by content using their hashed representation.

    Args:
        obj1 (Any): First object.
        obj2 (Any): Second object.

    Returns:
        bool: True if objects are equal, False otherwise.
    """
    return hash_obj(obj1) == hash_obj(obj2)


def unique_objects(
    obj_list: List[Any],
    return_idx: bool = False
) -> Union[List[Any], Tuple[List[Any], List[int]]]:
    """
    Deduplicate a list of objects (e.g., dicts/lists/strings), preserving order.

    Args:
        obj_list (List[Any]): A list of hashable or JSON-serializable objects.
        return_idx (bool): If True, also return the indices of first occurrences.

    Returns:
        Union[List[Any], Tuple[List[Any], List[int]]]: Unique elements, and optionally indices.

    Example:
        >>> unique_objects([{"x": 1}, {"x": 1}, [1, 2], "a", "a"])
        [{'x': 1}, [1, 2], 'a']
    """
    seen = set()
    uniques, indices = [], []

    for i, obj in enumerate(obj_list):
        obj_hash = hash_obj(obj)
        if obj_hash not in seen:
            seen.add(obj_hash)
            uniques.append(obj)
            indices.append(i)

    return (uniques, indices) if return_idx else uniques
