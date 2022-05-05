#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import expm
from scipy.spatial.distance import pdist, squareform
import os
import sys
from pathlib import Path



def loadCSVMatrix(filename):
	return np.loadtxt(filename,delimiter=",")


configFilename = "config.json"
argCount = len(sys.argv)
if(argCount > 1):
		configFilename = sys.argv[1]

outputDirectory = "output"


if(not os.path.exists(outputDirectory)):
		os.makedirs(outputDirectory)


with open(configFilename, "r") as fd:
		config = json.load(fd)




indexFilename = config["index"]
labelFilename = config["label"]
CSVDirectory = config["csv"]

with open(indexFilename, "r") as fd:
	indexData = json.load(fd)

with open(labelFilename, "r") as fd:
	labelData = json.load(fd)
	labelDataHasHeader = False

for entry in indexData:
	entryFilename = entry["filename"]
