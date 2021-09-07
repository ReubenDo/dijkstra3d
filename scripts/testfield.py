#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:18:20 2021

@author: reubendo
"""
import dijkstra3d
import numpy as np
import time

if __name__ == "__main__":
    # Reproducibility purposes
    np.random.seed(0)

    # 3D volume parameters
    shape = (42, 42, 20)
    spacing = [1., 0.5, 2.]

    # Tolerange
    epsilon = 1e-5

    # Creating image
    img = np.arange(np.prod(shape))
    np.random.shuffle(img)
    img = img.reshape(shape).astype(np.float32)

    # Sources (extreme points along the x axis)
    source = [4, 17, 9]
    target = [37, 9, 5]

    for l_grad, l_eucl in [[0, 1], [1, 0], [1, 1]]: # Test different configs
        for connectivity in [6, 18, 26]: # Test different connectivity
            # Length using path
            path, length_path = dijkstra3d.dijkstra(
                data=img,
                prob=np.zeros_like(img),
                source=source, 
                target=target,
                connectivity=connectivity, 
                spacing=spacing, 
                l_grad=l_grad, 
                l_eucl=l_eucl,
                l_prob=0.0,
                return_distance=True)

            # Length using Field
            field_dijkstra3d = dijkstra3d.distance_field(
                data=img,
                prob=np.zeros_like(img),
                source=source, 
                connectivity=connectivity, 
                spacing=spacing, 
                l_grad=l_grad, 
                l_eucl=l_eucl,
                l_prob=0.0)

            length_field = field_dijkstra3d[target[0],target[1],target[2]]

            relative_gap = abs(length_field-length_path)/length_path

            assert relative_gap<epsilon, f'{relative_gap} error with {l_grad}, {l_eucl}, {connectivity}, {source}'
    
    print("Test passed")
        