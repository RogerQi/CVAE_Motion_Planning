# Motion Planning with CVAE

Reproduction of B. Ichter, J. Harrison and M. Pavone, ”Learning Sampling Distributions
for Robot Motion Planning,” _2018 IEEE International Conference on
Robotics and Automation (ICRA)_, Brisbane, QLD, 2018, pp. 7087-7094.

Retrieved from https://arxiv.org/abs/1709.05448

##
To clone this repo:
```
git clone --recursive https://github.com/RogerQi/CVAE_Motion_Planning
```

TODOs
- Speed up collision detection
    - Use numpy vectorizaed Ops to support parallel configuration tests
    - BVH?
- Speed up sampling-based planning with spatial data structures
    - R-tree/Rebalancing KD-Tree/Ball tree