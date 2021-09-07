#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:18:20 2021

@author: reubendo
"""
import dijkstra3d
import scipy.sparse as spsp
import networkx as nx
import numpy as np
import SimpleITK as sitk 
import time

# Choose a edge weight function
def ewf( indsource, indtarget, edgelength2, l_grad, l_eucl):
    dist = l_eucl*edgelength2 + l_grad*abs(img.ravel()[indsource]-img.ravel()[indtarget])
    return dist

def s(a, b):
    return (a*a + b*b)**(1/2)

def create_graph(data, spacing, l_grad, l_eucl, connectivity=18):
    
    assert connectivity in (6, 18, 26), "connectivity should be in (6, 18, 26)"

    # Number of voxels
    N = np.prod(data.shape)

    # sparse addition may drop explicit zeros unless a slow dok format is used...
    # this should be easy to fix
    # discarding explicit zeros may lead to small holes
    # but that's fine for quick testing
    # spaddtsp = lambda a : a.todok() + a.transpose().todok()
    spaddtsp = lambda a : a + a.transpose()


    # Get basic indices to through the input image
    xind, yind, zind = np.mgrid[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]]

    # (x,y,z)-->(x+1,y,z)
    ind1 = np.ravel_multi_index([xind[:-1,:,:].ravel(), yind[:-1,:,:].ravel(), zind[:-1,:,:].ravel()], data.shape)
    ind2 = np.ravel_multi_index([xind[1:,:, :].ravel(), yind[1:,:,:].ravel(), zind[1:,:,:].ravel()], data.shape)
    dist = ewf(ind1, ind2, spacing[0], l_grad, l_eucl)
    bcm = spsp.coo_matrix((dist, (ind1, ind2)),shape=(N,N))
    # symmetrise this edge type
    bcm = spaddtsp(bcm)

    # (x,y,z)-->(x,y+1,z)
    ind1 = np.ravel_multi_index([xind[:,:-1,:].ravel(), yind[:,:-1,:].ravel(), zind[:,:-1,:].ravel()], data.shape)
    ind2 = np.ravel_multi_index([xind[:,1:,:].ravel(), yind[:,1:,:].ravel(), zind[:,1:,:].ravel()], data.shape)
    dist = ewf(ind1, ind2, spacing[1], l_grad, l_eucl)
    rcm = spsp.coo_matrix((dist, (ind1, ind2)),shape=(N,N))
    # symmetrise this edge type
    rcm = spaddtsp(rcm)

    # (x,y,z)-->(x,y,z+1)
    ind1 = np.ravel_multi_index([xind[:,:,:-1].ravel(), yind[:,:,:-1].ravel(), zind[:,:,:-1].ravel()], data.shape)
    ind2 = np.ravel_multi_index([xind[:,:,1:].ravel(), yind[:,:,1:].ravel(), zind[:,:,1:].ravel()], data.shape)
    dist = ewf(ind1, ind2, spacing[2], l_grad, l_eucl)
    lcm = spsp.coo_matrix((dist, (ind1, ind2)),shape=(N,N))
    # symmetrise this edge type
    lcm = spaddtsp(lcm)

    cm = bcm + rcm + lcm
    
    if connectivity>6:
        # diagonal (x,y,z)-->(x+1,y+1,z)
        ind1 = np.ravel_multi_index([xind[:-1,:-1,:].ravel(), yind[:-1,:-1,:].ravel(), zind[:-1,:-1,:].ravel()], data.shape)
        ind2 = np.ravel_multi_index([xind[1:,1:,:].ravel(), yind[1:,1:,:].ravel(), zind[1:,1:,:].ravel()], data.shape)
        dist = ewf(ind1, ind2, s(spacing[0], spacing[1]), l_grad, l_eucl)
        d1cm = spsp.coo_matrix((dist, (ind1, ind2)),shape=(N,N))
        # symmetrise this edge type
        d1cm = spaddtsp(d1cm)

        # diagonal (x,y,z)-->(x+1,y-1,z)
        ind1 = np.ravel_multi_index([xind[1:,:-1,:].ravel(), yind[1:,:-1,:].ravel(), zind[1:,:-1,:].ravel()], data.shape)
        ind2 = np.ravel_multi_index([xind[:-1,1:,:].ravel(), yind[:-1,1:,:].ravel(), zind[:-1,1:,:].ravel()], data.shape)
        dist = ewf(ind1, ind2, s(spacing[0], spacing[1]), l_grad, l_eucl)
        d2cm = spsp.coo_matrix((dist, (ind1, ind2)),shape=(N,N))
        # symmetrise this edge type
        d2cm = spaddtsp(d2cm)

        # diagonal (x,y,z)-->(x+1,y,z+1)
        ind1 = np.ravel_multi_index([xind[:-1,:,:-1].ravel(), yind[:-1,:,:-1].ravel(), zind[:-1,:,:-1].ravel()], data.shape)
        ind2 = np.ravel_multi_index([xind[1:,:,1:].ravel(), yind[1:,:,1:].ravel(), zind[1:,:,1:].ravel()], data.shape)
        dist = ewf(ind1, ind2, s(spacing[0], spacing[2]), l_grad, l_eucl)
        d1lm = spsp.coo_matrix((dist, (ind1, ind2)),shape=(N,N))
        # symmetrise this edge type
        d1lm = spaddtsp(d1lm)

        # diagonal (x,y,z)-->(x+1,y,z-1)
        ind1 = np.ravel_multi_index([xind[1:,:,:-1].ravel(), yind[1:,:,:-1].ravel(), zind[1:,:,:-1].ravel()], data.shape)
        ind2 = np.ravel_multi_index([xind[:-1,:,1:].ravel(), yind[:-1,:,1:].ravel(), zind[:-1,:,1:].ravel()], data.shape)
        dist = ewf(ind1, ind2, s(spacing[0], spacing[2]), l_grad, l_eucl)
        d2lm = spsp.coo_matrix((dist, (ind1, ind2)),shape=(N,N))
        # symmetrise this edge type
        d2lm = spaddtsp(d2lm)

        # diagonal (x,y,z)-->(x,y+1,z+1)
        ind1 = np.ravel_multi_index([xind[:,:-1,:-1].ravel(), yind[:,:-1,:-1].ravel(), zind[:,:-1,:-1].ravel()], data.shape)
        ind2 = np.ravel_multi_index([xind[:,1:,1:].ravel(), yind[:,1:,1:].ravel(), zind[:,1:,1:].ravel()], data.shape)
        dist = ewf(ind1, ind2, s(spacing[1], spacing[2]), l_grad, l_eucl)
        l1cm = spsp.coo_matrix((dist, (ind1, ind2)),shape=(N,N))
        # symmetrise this edge type
        l1cm = spaddtsp(l1cm)

        # diagonal (x,y,z)-->(x, y+1,z-1)
        ind1 = np.ravel_multi_index([xind[:,:-1,1:].ravel(), yind[:,:-1,1:].ravel(), zind[:,:-1,1:].ravel()], data.shape)
        ind2 = np.ravel_multi_index([xind[:,1:,:-1].ravel(), yind[:,1:,:-1].ravel(), zind[:,1:,:-1].ravel()], data.shape)
        dist = ewf(ind1, ind2, s(spacing[1], spacing[2]), l_grad, l_eucl)
        l2cm = spsp.coo_matrix((dist, (ind1, ind2)),shape=(N,N))
        # symmetrise this edge type
        l2cm = spaddtsp(l2cm)


        # 18 connectivity
        cm += d1cm + d2cm + d1lm + d2lm + l1cm + l2cm


    # Construct a networkx graph
    G = nx.from_scipy_sparse_matrix(cm,create_using=nx.Graph)

    return G

def distance_field_nx(G, seedind, shape):
    N = np.prod(shape)

    length = nx.multi_source_dijkstra_path_length(G, seedind, cutoff=np.inf)

    # Convert output from networkx into a numpy image
    k = np.array(list(length.keys()))
    v = np.array(list(length.values()))
    # sanity check the number of reached voxels
    assert N==k.shape[0], 'Error graph'
    distances = np.zeros(N)
    distances[k] = v
    dist_image = distances.reshape(shape)
    return dist_image
    


if __name__ == "__main__":
    # Reproducibility purposes
    np.random.seed(0)

    # 3D volume parameters
    shape = (42, 42, 20)
    spacing = [1., 1., 1.]

    # Tolerance
    epsilon = 1e-5

    # Creating image
    img = np.arange(np.prod(shape))
    np.random.shuffle(img)
    img = img.reshape(shape)

    # Sources (extreme points along the x axis)
    source1 = [4, 17, 9]
    source2 = [37, 9, 5]

    time_nx = []
    time_dijkstra = []
    for l_grad, l_eucl in [[0, 1], [1, 0], [1, 1]]: # Test different configs
        for connectivity in [6, 18]: # Test different connectivity
            for sources in [[source1], [source1, source2]]: # Test 1 or multiple source points
                # # Field using NX
                t = time.time()
                G = create_graph(img, spacing, l_grad, l_eucl, connectivity) 
                seedind = set(np.ravel_multi_index([[seed_position[k] for seed_position in sources] for k in range(3)], img.shape))
                field_nx = distance_field_nx(G, seedind, img.shape)
                time_nx.append(time.time() - t)

                # Field using dijkstra3d
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
                time_dijkstra.append(time.time() - t)

                relative_gap = abs(field_nx-field_dijkstra3d)/(field_nx+1e-6)

                assert  np.max(relative_gap)<epsilon, f'error with {l_grad}, {l_eucl}, {connectivity}, {sources}'
    
    print("Test passed")
    print(f"Mean time: NX {np.mean(time_nx)} - Dijkstra3d {np.mean(time_dijkstra)}")
        