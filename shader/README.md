## Optimizations

### Surface Area Heuristic (SAH)

When the bvh builder has a node full of triangles, it must divide them into group of two children. There are obvious approaches, like cutting the middle of the bounding box and put have the triangles in each side ignoring the shape of what's inside. These approaches often produce child boxes that are large and overlapping.

Suppose that a ray is known to hit a parent box. What is the chance it will also hit a smaller box inside it? SAH estimates the likelihood that a ray will hit a surface base on how large it is relative to the area of it's parent's box. The intuition is that a box's probability of getting hit by a ray is proportional to its surface area relative to the parent's.

The formula rewards small boxes sinces few rays reach them and balances counts since surface area becomes the basis or partition rather than count.

### Splitting with binning

Testing every possible split position exactly would be slow, so the code approximates:

1. Pick an axis. It uses the longest axis of the triangles’ centers.
2. Sort into bins. It divides that axis into 12 equal bins (kSahBuckets) and drops each triangle into a bin by the position of its center. Each bin tracks a triangle count and a bounding box.
3. Sweep both ways. One pass from the left builds up the area and count for “bins 0 through s.” A pass from the right does the same for “bins $s+1$ through 11.” That’s what the `leftArea`, `leftCount`, `rightArea` and `rightCount` arrays hold.
4. Score the gaps. It evaluates the cost formula at each of the 11 boundaries between bins and keeps the cheapest.
5. Partition. std::partition reorders the triangle indices so the left group comes first, then the builder recurses into each half.`build` and `traverse` suffer from low occupancy rates. Each thread must know whether the parent node contains the triangle.

### Why This Doesn't Translate to HPC

With GPUs we are primarily concerned with thread occupancy. With SAH, the relationship between parent an child node is preserved. We have to finish processing the root before we know which ranges its childrens cover. Regardless of whether BFS or recursive DFS is used, the higher levels of the tree will run at low occupancy.
