#!/bin/bash

# predict equalised single cell-lines with a gradient boosting classifier

while read CELL_LINE; do
    python equalise_groups.py "$CELL_LINE" &
done < cell_lines.txt