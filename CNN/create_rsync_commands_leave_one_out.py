#!/usr/bin/env python

import sys
import os

assert len(sys.argv) >= 2

cell_line_to_rm = sys.argv[1]

cell_lines = ["MDA-231",
              "MDA-157",
              "MCF7",
              "KPL4",
              "SKBR3",
              "T47D",
              "HCC1954",
              "HCC1569"]

if cell_line_to_rm not in cell_lines:
    raise ValueError("unknwown cell-line, options are :{}".format(cell_lines))

cell_lines.remove(cell_line_to_rm)

EXISTING_DIR = "/exports/igmm/datastore/Drug-Discovery/scott/2018-04-24_nncell_data_300_{}/"

NEW_DIR_NAME = "/exports/eddie/scratch/s1027820/chopped_array/data_excluding_{}/"

for cell_line in cell_lines:
    cmd = "rsync -a " + EXISTING_DIR.format(cell_line) + " " + NEW_DIR_NAME.format(cell_line_to_rm)
    sys.stdout.write(cmd + "\n")
