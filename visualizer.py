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
    
    scale = 80 / n
    
    highlightCost1, highlightCost2 = 100, 200
    
    blkMark, costMark1, costMark2 = 0, 0, 0
    for i in range(n):
        for j in range(n):
            if graph[i][j] == -1:
                if blkMark == 0:
                    plt.fill([i - 0.5, i - 0.5, i + 0.5, i + 0.5], [j - 0.5, j + 0.5, j + 0.5, j - 0.5], "k", label = 'blockage')
                    #plt.plot(i, j, "k.", markersize = 5 * scale, label = 'blockage')
                    blkMark = 1
                else:
                    plt.fill([i - 0.5, i - 0.5, i + 0.5, i + 0.5], [j - 0.5, j + 0.5, j + 0.5, j - 0.5], "k")
                    #plt.plot(i, j, "k.", markersize = 5 * scale)
            elif graph[i][j] > highlightCost2:
                if costMark1 == 0:
                    plt.fill([i - 0.5, i - 0.5, i + 0.5, i + 0.5], [j - 0.5, j + 0.5, j + 0.5, j - 0.5], "y", label = 'cost > ' + str(highlightCost2))
                    costMark1 = 1
                else:
                    plt.fill([i - 0.5, i - 0.5, i + 0.5, i + 0.5], [j - 0.5, j + 0.5, j + 0.5, j - 0.5], "y")
            elif graph[i][j] > highlightCost1:
                if costMark2 == 0:
                    plt.fill([i - 0.5, i - 0.5, i + 0.5, i + 0.5], [j - 0.5, j + 0.5, j + 0.5, j - 0.5], "m", label = str(highlightCost1) + "<=cost<" + str(highlightCost2));
                    costMark2 = 1
                else:
                    plt.fill([i - 0.5, i - 0.5, i + 0.5, i + 0.5], [j - 0.5, j + 0.5, j + 0.5, j - 0.5], "m")
    
    numPins = int(f.readline())
    for i in range(numPins):
        x, y = map(int, f.readline().split())
        if i == 0:
            plt.plot(x, y, "g.", markersize = 20 * scale, alpha = 0.2, label = "pin")
        else:
            plt.plot(x, y, "g.", markersize = 20 * scale, alpha = 0.2)
            
    dx, dy = [-1, 1, 0, 0], [0, 0, -1, 1]
    
    gpu, gpuMark = int(f.readline()), 0
    gpuPath = {}
    for i in range(gpu):
        x, y = map(int, f.readline().split())
        gpuPath[x, y] = 1
    for e in gpuPath:
        x, y = e
        for d in range(4):
            if gpuPath.get((x + dx[d], y + dy[d])) != None:
                if gpuMark == 0:
                    plt.plot([x, x + dx[d]], [y, y + dy[d]], "r", linewidth = 0.2, alpha = 0.2, label = "gpu route")
                    gpuMark = 1
                else:
                    plt.plot([x, x + dx[d]], [y, y + dy[d]], "r", linewidth = 0.2, alpha = 0.2);
            
            #plt.plot(x, y , "r.", markersize = 5 * scale, alpha = 0.4, label = "gpu route")
    cpu, cpuMark = int(f.readline()), 0
    cpuPath = {}
    for i in range(cpu):
        x, y = map(int, f.readline().split())
        cpuPath[x, y] = 1
    for e in cpuPath:
        x, y = e
        for d in range(4):
            if cpuPath.get((x + dx[d], y + dy[d])) != None:
                if cpuMark == 0:
                    plt.plot([x, x + dx[d]], [y, y + dy[d]], "b", linewidth = 0.2, alpha = 0.2, label = "cpu route")
                    cpuMark = 1
                else:
                    plt.plot([x, x + dx[d]], [y, y + dy[d]], "b", linewidth = 0.2, alpha = 0.2);
    plt.legend()
    plt.title(args.input_file)
    plt.savefig("output/" + args.input_file + ".pdf");
    