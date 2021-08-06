#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:41:20 2021

@author: reubendo
"""
import dijkstra3d
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import os

    
if __name__ == "__main__":
    # Reproducibility purposes
    np.random.seed(0)

    connectivity = 26
    l_eucl = 1
    l_grad = 1
    spacing = [1, 0.5, 2]

    
    for N_seed in [1,10, 100, 1000]:
        times = []
        for shape in tqdm([(k,k,k) for k in range(10,210,10)]):

            # Creating image
            img = np.arange(np.prod(shape))
            np.random.shuffle(img)
            img = img.reshape(shape)

            N_seed = np.min([N_seed,np.prod(shape)-1])

            # Sources (extreme points along the x axis)
            sourcesind = np.random.choice(np.arange(np.prod(shape)), size=N_seed, replace=False)
            sources = np.unravel_index(sourcesind, shape)
            sources = np.stack(sources,1).tolist()

            
            t = time.time()
            field_dijkstra3d = dijkstra3d.distance_field(
                data=img,
                prob=np.zeros_like(img),
                source=sources, 
                connectivity=connectivity, 
                spacing=spacing, 
                l_grad=l_grad, 
                l_eucl=l_eucl,
                l_prob=0.0)
            times.append(time.time() - t)

                        
        list_x = [np.prod((k,k,k)) for k in range(10,210,10)]  
        plt.figure()  
        plt.title(f'Time in (s) per number of voxels - Seeds: {N_seed} random points')
        plt.xlabel('Spatial size')
        plt.ylabel('Execution time (seconds)')
        plt.scatter(list_x, times, marker = 'o', edgecolors = 'none')
        plt.tight_layout()
        plt.savefig(os.path.join('figures',  f'speed_{N_seed}.png'))
        