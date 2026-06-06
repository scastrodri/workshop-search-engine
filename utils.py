from typing import Sequence, List, TypeVar

T = TypeVar('T')

def make_batches(seq: Sequence[T], n: int) -> List[List[T]]:
    """
    Split a sequence into batches of size n.

    Args:
        seq: Any sequence (list, array, tuple, etc.)
        n: Batch size (must be > 0)

    Returns:
        List of batches, each of size n (last batch may be smaller)
    """
    if n <= 0:
        raise ValueError("The batch size 'n' must be greater than 0.")
        
    return [seq[i:i + n] for i in range(0, len(seq), n)]

if __name__ == "__main__":
    sample = list(range(10))
    batches = make_batches(sample, 3)
    assert len(batches) == 4
    assert batches[0] == [0, 1, 2]
    assert batches[-1] == [9]
    print("All checks passed.")