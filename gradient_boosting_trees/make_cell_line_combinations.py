"""
author: Scott Warchal
date: 2018-05-15

make combinations of cell-lines for cumulative training
# NOTE: not working properly, will have to manually edit the output
"""

import itertools

CELL_LINES = ["MDA-231",
              "MDA-157",
              "MCF7",
              "KPL4",
              "T47D",
              "SKBR3",
              "HCC1954",
              "HCC1569"]

def all_combinations(iterable):
    """
    Returns all non-repeating combinations
    of elements in `iterable`. Where the number
    of combinations is 2^len(iterable) - 1

    Returns:
    --------
    a list of lists
    """
    result = []
    for n in range(0, len(iterable) + 1):
        for subset in itertools.combinations(iterable, n):
            result.append(list(subset))
    return result[1:] # skip first element as it's empty


def main():
    combinations = all_combinations(CELL_LINES)
    for comb in combinations:
        print(*comb)


if __name__ == "__main__":
    main()

