
# dijkstra3d for InExtremIS
Dijkstra's Shortest Path variants for 6, 18, and 26-connected 3D Image Volumes or 4 and 8-connected 2D images. 

Code based on [dijkstra3d](https://github.com/seung-lab/dijkstra3d).

Let be given a path between a and b (x0 = a, x1, ··· , xn−1, xn = b). The length of the path is:
![image](https://user-images.githubusercontent.com/17268715/128386608-5c15bb25-cafc-4ca5-ba32-9e44cc3b9421.png)
where p(xk) is the pseudo-probability that the voxel xk is part of the background.

```python
import numpy as np
import SimpleITK as sitk 
import dijkstra3d


# Test image and manual extreme points
input_name = "data/vs_gk_1_t2.nii.gz"
input_extreme = "data/vs_gk_1_extremepoints.nii.gz"
input_pred = "data/vs_gk_1_pred.nii.gz"

img = sitk.GetArrayFromImage(sitk.ReadImage(input_name)).transpose() 
extreme = sitk.GetArrayFromImage(sitk.ReadImage(input_extreme)).transpose()                              
prob_background = sitk.GetArrayFromImage(sitk.ReadImage(input_pred)).transpose()

spacing = sitk.ReadImage(input_name).GetSpacing()
shape = img.shape

# Source and target points (extreme points along the x axis)
source = np.argwhere(extreme==1).squeeze().tolist()
target = np.argwhere(extreme==2).squeeze().tolist()

# Normalization factors used in InExtremIS
l_eucl = 1.0 / np.max([spacing[k]*shape[k] for k in range(3)])
max_grad = np.max([abs(np.diff(img,axis=k)).max() for k in range(3)])
l_grad = 1.0 / max_grad
l_prob = 1.0

path = dijkstra3d.dijkstra(
    data=img,
    prob=prob_background,
    source=source, 
    target=target,
    connectivity=26, 
    spacing=spacing, 
    l_grad=l_grad, 
    l_eucl=l_eucl,
    l_prob=l_prob)

```

Perform dijkstra's shortest path algorithm on a 3D image grid.  


## Python Direct Installation

*Requires a C++ compiler.*

```bash
git clone https://github.com/seung-lab/dijkstra3d.git
cd dijkstra3d
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py develop
```

## Voxel Connectivity Graph

You may optionally provide a unsigned 32-bit integer image that specifies the allowed directions of travel per voxel as a directed graph. Each voxel in the graph contains a bitfield of which only the lower 26 bits are used to specify allowed directions. The top 6 bits have no assigned meaning. It is possible to use smaller width bitfields for 2D images (uint8) or for undirected graphs (uint16), but they are not currently supported. Please open an Issue or Pull Request if you need this functionality.

The specification below shows the meaning assigned to each bit. Bit 32 is the MSB, bit 1 is the LSB. Ones are allowed directions and zeros are disallowed directions.

```
    32     31     30     29     28     27     26     25     24     23     
------ ------ ------ ------ ------ ------ ------ ------ ------ ------
unused unused unused unused unused unused -x-y-z  x-y-z -x+y-z +x+y-z

    22     21     20     19     18     17     16     15     14     13
------ ------ ------ ------ ------ ------ ------ ------ ------ ------
-x-y+z +x-y+z -x+y+z    xyz   -y-z    y-z   -x-z    x-z    -yz     yz

    12     11     10      9      8      7      6      5      4      3
------ ------ ------ ------ ------ ------ ------ ------ ------ ------
   -xz     xz   -x-y    x-y    -xy     xy     -z     +z     -y     +y  
     2      1
------ ------
    -x     +x
```

There is an assistive tool available for producing these graphs from adjacent labels in the [cc3d library](https://github.com/seung-lab/connected-components-3d).


### What is that pairing_heap.hpp?

Early on, I anticipated using decrease key in my heap and implemented a pairing heap, which is supposed to be an improvement on the Fibbonacci heap. However, I ended up not using decrease key, and the STL priority queue ended up being faster. If you need a pairing heap outside of boost, check it out.

## References

1. E. W. Dijkstra. "A Note on Two Problems in Connexion with Graphs" Numerische Mathematik 1. pp. 269-271. (1959)  
2. E. W. Dijkstra. "Go To Statement Considered Harmful". Communications of the ACM. Vol. 11, No. 3, pp. 147-148. (1968)
3. Pohl, Ira. "Bi-directional Search", in Meltzer, Bernard; Michie, Donald (eds.), Machine Intelligence, 6, Edinburgh University Press, pp. 127-140. (1971)
