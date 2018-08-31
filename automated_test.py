import dijkstra
import numpy as np

TEST_TYPES = (
  np.float32, np.float64,
  np.uint64, np.uint32, np.uint16, np.uint8,
  np.int64, np.int32, np.int16, np.int8,
  np.bool
)

def test_dijkstra2d_10x10():
  for dtype in TEST_TYPES:
    values = np.ones((10,10,1), dtype=dtype)
    
    path = dijkstra.dijkstra(values, (0,0,0), (3,0,0))

    assert len(path) == 4
    assert np.all(path == np.array([
      [0,0,0],
      [1,1,0],
      [2,1,0],
      [3,0,0],
    ]))

    path = dijkstra.dijkstra(values, (0,0,0), (5,5,0))

    assert len(path) == 6
    assert np.all(path == np.array([
      [0,0,0],
      [1,1,0],
      [2,2,0],
      [3,3,0],
      [4,4,0],
      [5,5,0],
    ]))

    path = dijkstra.dijkstra(values, (0,0,0), (9,9,0))
    
    assert len(path) == 10
    assert np.all(path == np.array([
      [0,0,0],
      [1,1,0],
      [2,2,0],
      [3,3,0],
      [4,4,0],
      [5,5,0],
      [6,6,0],
      [7,7,0],
      [8,8,0],
      [9,9,0]
    ]))

def test_dijkstra2d_10x10_off_origin():
  for dtype in TEST_TYPES:
    values = np.ones((10,10,1), dtype=dtype)
    
    path = dijkstra.dijkstra(values, (2,0,0), (3,0,0))

    assert len(path) == 2
    assert np.all(path == np.array([
      [2,0,0],
      [3,0,0],
    ]))

    path = dijkstra.dijkstra(values, (2,1,0), (3,0,0))

    assert len(path) == 2
    assert np.all(path == np.array([
      [2,1,0],
      [3,0,0],
    ]))

    path = dijkstra.dijkstra(values, (9,9,0), (5,5,0))

    assert len(path) == 5
    assert np.all(path == np.array([
      [9,9,0],
      [8,8,0],
      [7,7,0],
      [6,6,0],
      [5,5,0],
    ]))

def test_dijkstra3d_3x3x3():
  for dtype in TEST_TYPES:
    values = np.ones((3,3,3), dtype=dtype)

    path = dijkstra.dijkstra(values, (0,0,0), (2,2,2))
    assert np.all(path == np.array([
      [0,0,0],
      [1,1,1],
      [2,2,2]
    ]))

    path = dijkstra.dijkstra(values, (2,2,2), (0,0,0))
    assert np.all(path == np.array([
      [2,2,2],
      [1,1,1],
      [0,0,0]
    ]))

def test_distance_field_2d():
  for dtype in TEST_TYPES:
    values = np.ones((5,5), dtype=dtype)
    
    field = dijkstra.distance_field(values, (0,0))

    assert np.all(field == np.array([
      [
        [0, 1, 2, 3, 4],
        [1, 1, 2, 3, 4],
        [2, 2, 2, 3, 4],
        [3, 3, 3, 3, 4],
        [4, 4, 4, 4, 4],
      ]
    ]))

    field = dijkstra.distance_field(values, (4,4))

    assert np.all(field == np.array([
      [
        [4, 4, 4, 4, 4],
        [4, 3, 3, 3, 3],
        [4, 3, 2, 2, 2],
        [4, 3, 2, 1, 1],
        [4, 3, 2, 1, 0],
      ]
    ]))

    field = dijkstra.distance_field(values, (2,2))

    assert np.all(field == np.array([
      [
        [2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2],
        [2, 1, 0, 1, 2],
        [2, 1, 1, 1, 2],
        [2, 2, 2, 2, 2],
      ]
    ]))


    field = dijkstra.distance_field(values * 2, (2,2))

    assert np.all(field == np.array([
      [
        [4, 4, 4, 4, 4],
        [4, 2, 2, 2, 4],
        [4, 2, 0, 2, 4],
        [4, 2, 2, 2, 4],
        [4, 4, 4, 4, 4],
      ]
    ]))
