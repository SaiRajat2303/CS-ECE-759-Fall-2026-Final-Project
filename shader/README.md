### Surface Area Heuristic(SAH)
When the bvh builder has anode full of triangles, it must divide them into group of two children. There are obvious approaches, like cutting the middle of the bounding box and put have the triangles in each side ignoring the shape of what's inside. That often produces child boxes that are large and overlapping. 

The SAH estimates how much each candidate split would cost a ray. 

The intuition is that a box's probability of getting hit by a ray is proportional to its surface area as a ratio of the parent