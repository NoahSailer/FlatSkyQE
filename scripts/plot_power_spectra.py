import numpy as np
import matplotlib.pyplot as plt

lCen = np.genfromtxt('lCen.txt')
lEdges = np.genfromtxt('lEdges.txt')
mean = np.genfromtxt('mean_power.txt')
std = np.genfromtxt('std_power.txt')
thy = np.genfromtxt('theory_power.txt')

plt.errorbar(lCen,mean/thy,yerr=std/thy,lw=2,ls='')
plt.show()
