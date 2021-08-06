"""
Cython binding for C++ dijkstra's shortest path algorithm
applied to 3D images. 

Contains:
  dijkstra - Find the shortest 26-connected path from source
    to target using the values of each voxel as edge weights.\

  parental_field / path_from_parents - Same as dijkstra,
    but if you're computing dijkstra multiple times on
    the same image, this can be much much faster.


Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018-February 2020
"""

from libc.stdlib cimport calloc, free
from libc.stdint cimport (
   int8_t,  int16_t,  int32_t,  int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t
)
from cpython cimport array 
import array
import sys

from libcpp.vector cimport vector
cimport numpy as cnp
import numpy as np

__VERSION__ = '1.9.2'

ctypedef fused UINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t

class DimensionError(Exception):
  pass

cdef extern from "dijkstra3d.hpp" namespace "dijkstra":
  cdef vector[OUT] dijkstra3d[T,OUT](
    T* field, 
    float* prob,
    size_t sx, size_t sy, size_t sz, 
    size_t source, size_t target,
    float dx, float dy, float dz,
    float w_grad, float w_eucl, float w_prob,
    int connectivity,
    uint32_t* voxel_graph
  )
  cdef vector[T] query_shortest_path[T](
    T* parents, T target
  ) 
  cdef float* distance_field3d[T](
    T* field,
    float* prob,
    size_t sx, size_t sy, size_t sz, 
    vector[size_t] source,
    float dx, float dy, float dz,
    float w_grad, float w_eucl, float w_prob,
    int connectivity,
    uint32_t* voxel_graph
  )
  
def format_voxel_graph(voxel_graph):
  while voxel_graph.ndim < 3:
    voxel_graph = voxel_graph[..., np.newaxis]

  if not np.issubdtype(voxel_graph.dtype, np.uint32):
    voxel_graph = voxel_graph.astype(np.uint32, order="F")
  
  return np.asfortranarray(voxel_graph)

def dijkstra(
  data, prob, source, target, connectivity=26, 
  spacing=(1,1,1), l_grad=0.0, l_eucl=1.0, l_prob=0.0,
  voxel_graph=None
):
  """
  Perform dijkstra's shortest path algorithm
  on a 3D image grid. Vertices are voxels and
  edges are the 26 nearest neighbors (except
  for the edges of the image where the number
  of edges is reduced).
  
  For given input voxels A and B, the edge
  weight from A to B is B and from B to A is
  A. All weights must be non-negative (incl. 
  negative zero).
  
  Parameters:
   Data: Input weights in a 2D or 3D numpy array. 
   source: (x,y,z) coordinate of starting voxel
   target: (x,y,z) coordinate of target voxel
  
  Returns: 1D numpy array containing indices of the path from
    source to target including source and target.
  """
  dims = len(data.shape)
  if dims not in (2,3):
    raise DimensionError("Only 2D and 3D image sources are supported. Got: " + str(dims))

  assert data.shape==prob.shape, "Probability map and Image must have the same shape"

  if dims == 2:
    if connectivity == 4:
      connectivity = 6
    elif connectivity == 8:
      connectivity = 18 # or 26 but 18 might be faster

  if connectivity not in (6, 18, 26):
    raise ValueError(
      "Only 6, 18, and 26 connectivities are supported. Got: " + str(connectivity)
    )

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.uint32, order='F')

  _validate_coord(data, source)
  _validate_coord(data, target)

  if dims == 2:
    data = data[:, :, np.newaxis]
    prob = prob[:, :, np.newaxis]
    source = list(source) + [ 0 ]
    target = list(target) + [ 0 ]
    print(prob.shape)

  if voxel_graph is not None:
    voxel_graph = format_voxel_graph(voxel_graph)

  data = np.asfortranarray(data)
  prob = np.asfortranarray(prob)

  cdef size_t cols = data.shape[0]
  cdef size_t rows = data.shape[1]
  cdef size_t depth = data.shape[2]

  path = _execute_dijkstra(
    data, prob, source, target, connectivity, spacing,
    l_grad, l_eucl, l_prob
  )

  return _path_to_point_cloud(path, dims, rows, cols)

def _validate_coord(data, coord):
  dims = len(data.shape)

  if len(coord) != dims:
    raise IndexError(
      "Coordinates must have the same dimension as the data. coord: {}, data shape: {}"
        .format(coord, data.shape)
    )

  for i, size in enumerate(data.shape):
    if coord[i] < 0 or coord[i] >= size:
      raise IndexError("Selected voxel {} was not located inside the array.".format(coord))

def _path_to_point_cloud(path, dims, rows, cols):
  ptlist = np.zeros((path.shape[0], dims), dtype=np.uint32)

  cdef size_t sxy = rows * cols
  cdef size_t i = 0

  if dims == 3:
    for i, pt in enumerate(path):
      ptlist[ i, 0 ] = pt % cols
      ptlist[ i, 1 ] = (pt % sxy) / cols
      ptlist[ i, 2 ] = pt / sxy
  else:
    for i, pt in enumerate(path):
      ptlist[ i, 0 ] = pt % cols
      ptlist[ i, 1 ] = (pt % sxy) / cols

  return ptlist


def _execute_dijkstra(
  data, prob, source, target, int connectivity, 
  spacing=(1,1,1), l_grad=0.0, l_eucl=1.0, l_prob=0.0,
  voxel_graph=None
):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef float[:,:,:] prob_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef uint32_t[:,:,:] voxel_graph_memview
  cdef uint32_t* voxel_graph_ptr = NULL
  if voxel_graph is not None:
    voxel_graph_memview = voxel_graph
    voxel_graph_ptr = <uint32_t*>&voxel_graph_memview[0,0,0]

  cdef size_t sx = data.shape[0]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[2]

  cdef float dx = float(spacing[0])
  cdef float dy = float(spacing[1])
  cdef float dz = float(spacing[2])

  cdef float w_grad = float(l_grad)
  cdef float w_eucl = float(l_eucl)
  cdef float w_prob = float(l_prob)

  cdef size_t src = source[0] + sx * (source[1] + sy * source[2])
  cdef size_t sink = target[0] + sx * (target[1] + sy * target[2])

  cdef vector[uint32_t] output32
  cdef vector[uint64_t] output64

  sixtyfourbit = data.size > np.iinfo(np.uint32).max

  # data = np.asarray(data, np.float32)
  prob = np.asarray(prob, np.float32)

  prob_memviewfloat = prob
  
  dtype = data.dtype

  if dtype == np.float32:
    arr_memviewfloat = data
    if sixtyfourbit:
      output64 = dijkstra3d[float, uint64_t](
        &arr_memviewfloat[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink,
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = dijkstra3d[float, uint32_t](
        &arr_memviewfloat[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink,
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
  elif dtype == np.float64:
    arr_memviewdouble = data
    if sixtyfourbit:
      output64 = dijkstra3d[double, uint64_t](
        &arr_memviewdouble[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink, 
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = dijkstra3d[double, uint32_t](
        &arr_memviewdouble[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink, 
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    if sixtyfourbit:
      output64 = dijkstra3d[uint64_t, uint64_t](
        &arr_memview64[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink, 
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = dijkstra3d[uint64_t, uint32_t](
        &arr_memview64[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink, 
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int32, np.uint32):
    arr_memview32 = data.astype(np.uint32)
    if sixtyfourbit:
      output64 = dijkstra3d[uint32_t, uint64_t](
        &arr_memview32[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink, 
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = dijkstra3d[uint32_t, uint32_t](
        &arr_memview32[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink, 
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    if sixtyfourbit:
      output64 = dijkstra3d[uint16_t, uint64_t](
        &arr_memview16[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink, 
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = dijkstra3d[uint16_t, uint32_t](
        &arr_memview16[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink, 
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int8, np.uint8, bool):
    arr_memview8 = data.astype(np.uint8)
    if sixtyfourbit:
      output64 = dijkstra3d[uint8_t, uint64_t](
        &arr_memview8[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink, 
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = dijkstra3d[uint8_t, uint32_t](
        &arr_memview8[0,0,0],
        &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src, sink, 
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )

  cdef uint32_t* output_ptr32
  cdef uint64_t* output_ptr64

  cdef uint32_t[:] vec_view32
  cdef uint64_t[:] vec_view64

  if sixtyfourbit:
    output_ptr64 = <uint64_t*>&output64[0]
    if output64.size() == 0:
      return np.zeros((0,), dtype=np.uint64)
    vec_view64 = <uint64_t[:output64.size()]>output_ptr64
    buf = bytearray(vec_view64[:])
    output = np.frombuffer(buf, dtype=np.uint64)
  else:
    output_ptr32 = <uint32_t*>&output32[0]
    if output32.size() == 0:
      return np.zeros((0,), dtype=np.uint32)
    vec_view32 = <uint32_t[:output32.size()]>output_ptr32
    buf = bytearray(vec_view32[:])
    output = np.frombuffer(buf, dtype=np.uint32)

  return output[::-1]


def _execute_distance_field(data, prob, sources, connectivity, spacing, l_grad, l_eucl, l_prob, voxel_graph):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble
  cdef float[:,:,:] prob_memviewfloat

  cdef uint32_t[:,:,:] voxel_graph_memview
  cdef uint32_t* voxel_graph_ptr = NULL
  if voxel_graph is not None:
    voxel_graph_memview = voxel_graph
    voxel_graph_ptr = <uint32_t*>&voxel_graph_memview[0,0,0]

  cdef size_t sx = data.shape[0]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[2]

  cdef float dx = float(spacing[0])
  cdef float dy = float(spacing[1])
  cdef float dz = float(spacing[2])

  cdef float w_grad = float(l_grad)
  cdef float w_eucl = float(l_eucl)
  cdef float w_prob = float(l_prob)

  prob = np.asarray(prob, np.float32)
  prob_memviewfloat = prob

  cdef vector[size_t] src
  for source in sources:
    src.push_back(source[0] + sx * (source[1] + sy * source[2]))

  cdef float* dist

  dtype = data.dtype

  if dtype == np.float32:
    arr_memviewfloat = data
    dist = distance_field3d[float](
      &arr_memviewfloat[0,0,0],
      &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src,  
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )

  elif dtype == np.float64:
    arr_memviewdouble = data
    dist = distance_field3d[double](
      &arr_memviewdouble[0,0,0],
      &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src,  
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    dist = distance_field3d[uint64_t](
      &arr_memview64[0,0,0],
      &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src,  
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    dist = distance_field3d[uint32_t](
      &arr_memview32[0,0,0],
      &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src,  
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    dist = distance_field3d[uint16_t](
      &arr_memview16[0,0,0],
      &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src,  
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int8, np.uint8, bool):
    arr_memview8 = data.astype(np.uint8)
    dist = distance_field3d[uint8_t](
      &arr_memview8[0,0,0],
      &prob_memviewfloat[0,0,0],
        sx, sy, sz,
        src,  
        dx, dy, dz,
        w_grad, w_eucl, w_prob,
        connectivity,
        voxel_graph_ptr
      )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  cdef size_t voxels = sx * sy * sz
  cdef float[:] dist_view = <float[:voxels]>dist

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(dist_view[:])
  free(dist)
  # I don't actually understand why order F works, but it does.
  return np.frombuffer(buf, dtype=np.float32).reshape(data.shape, order='F')


def distance_field(data, prob, source, connectivity=26, spacing=(1,1,1), l_grad=0.0, l_eucl=1.0, l_prob=0.0, voxel_graph=None):
  """
  Use dijkstra's shortest path algorithm
  on a 3D image grid to generate a weighted 
  distance field from one or more source voxels. Vertices are 
  voxels and edges are the 26 nearest neighbors 
  (except for the edges of the image where 
  the number of edges is reduced).
  
  For given input voxels A and B, the edge
  weight from A to B is B and from B to A is
  A. All weights must be non-negative (incl. 
  negative zero).
  
  Parameters:
   data: Input weights in a 2D or 3D numpy array. 
   source: (x,y,z) coordinate or list of coordinates 
    of starting voxels.
   connectivity: 26, 18, or 6 connected.
  
  Returns: 2D or 3D numpy array with each index
    containing its distance from the source voxel.
  """
  dims = len(data.shape)
  if dims not in (2,3):
    raise DimensionError("Only 2D and 3D image sources are supported. Got: " + str(dims))

  assert data.shape==prob.shape, "Probability map and Image must have the same shape"

  if dims == 2:
    if connectivity == 4:
      connectivity = 6
    elif connectivity == 8:
      connectivity = 18 # or 26 but 18 might be faster

  if connectivity not in (6, 18, 26):
    raise ValueError(
      "Only 6, 18, and 26 connectivities are supported. Got: " + str(connectivity)
    )
  
  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.float32)

  source = np.array(source, dtype=np.uint64)
  if source.ndim == 1:
    source = source[np.newaxis, :]
  for src in source:
    _validate_coord(data, src)
  if source.shape[1] < 3:
    tmp = np.zeros((source.shape[0], 3), dtype=np.uint64)
    tmp[:, :source.shape[1]] = source[:,:]
    source = tmp
  while data.ndim < 3:
    data = data[..., np.newaxis]
    prob = prob[..., np.newaxis]

  if voxel_graph is not None:
    voxel_graph = format_voxel_graph(voxel_graph)

  data = np.asfortranarray(data)

  field = _execute_distance_field(data, prob, source, connectivity, spacing, l_grad, l_eucl, l_prob, voxel_graph)
  if dims < 3:
    field = np.squeeze(field, axis=2)
  if dims < 2:
    field = np.squeeze(field, axis=1)

  return field