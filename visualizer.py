import matplotlib.pyplot as plt
import argparse
import time, os, sys
import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', default='case6')
args = parser.parse_args()

with open("output/" + args.input_file + "output.txt", 'r') as f:
    n = int(f.readline())
    graph = []
    
    plt.plot([0, 0], [0, 0], '.', alpha = 0)
    plt.plot([n, n], [n, n], '.', alpha = 0)
    
    for i in range(n):
        graph.append([int(x) for x in f.readline().split()])
        
    mark = 0
    for i in range(n):
        for j in range(n):
            if graph[i][j] == -1:
                if mark == 0:
                    plt.plot(i, j, "k.", markersize = 5, label = 'blockage')
                    mark = 1
                else:
                    plt.plot(i, j, "k.", markersize = 5)
    
    numPins = int(f.readline())
    for i in range(numPins):
        x, y = map(int, f.readline().split())
        if i == 0:
            plt.plot(x, y, "g.", markersize = 20, alpha = 0.2, label = "pin")
        else:
            plt.plot(x, y, "g.", markersize = 20, alpha = 0.2)
    gpu = int(f.readline())
    for i in range(gpu):
        x, y = map(int, f.readline().split())
        if i == 0:
            plt.plot(x, y , "r.", markersize = 5, alpha = 0.4, label = "gpu route")
        else:
            plt.plot(x, y , "r.", markersize = 5, alpha = 0.4)
    cpu = int(f.readline())
    for i in range(cpu):
        x, y = map(int, f.readline().split())
        if i == 0:
            plt.plot(x, y, "b.", markersize = 5, alpha = 0.4, label = "cpu route")
        else:
            plt.plot(x, y, "b.", markersize = 5, alpha = 0.4)
    plt.legend()
    plt.savefig("output/" + args.input_file + ".pdf");
    