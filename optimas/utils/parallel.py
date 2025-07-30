from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback
from typing import Callable, List, Any, Tuple, Optional
from optimas.utils.logger import setup_logger

logger = setup_logger(__name__)

def run_parallel_tasks(
    task_func: Callable[..., Any],
    task_args: List[Tuple[Any, ...]],
    max_workers: int = 4,
    task_desc: str = "[parallel processing]",
    on_error: Optional[Callable[[Tuple[Any, ...], Exception], None]] = None,
) -> List[Any]:
    """
    Run `task_func` in parallel across multiple argument sets using threads.

    Args:
        task_func (Callable): Function to run in parallel. Each call gets a tuple of args.
        task_args (List[Tuple]): A list of argument tuples, one per call.
        max_workers (int): Max number of threads.
        use_tqdm (bool): Whether to show a progress bar.
        task_desc (str): Description for tqdm progress bar.
        on_error (Callable): Optional function to call on exception with (args, exception).

    Returns:
        List[Any]: List of results in completion order (not necessarily input order).
                   Failed results are returned as None.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task_func, *args): args for args in task_args}
        iterator = tqdm(as_completed(futures), total=len(futures), desc=task_desc)

        for future in iterator:
            args = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.warning(f"Task failed with args {args}:\n{traceback.format_exc()}")
                if on_error:
                    on_error(args, e)
                results.append(None)

    return results
