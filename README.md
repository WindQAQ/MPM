# MPM Simulation & Voxels Rendering

![snow balls collision](images/two_snowballs_p6146_g200.png)

We try to simulate fluid-like materials, such as snow and sand, with material point method. MPM is implemented in C++ with CUDA support. And OpenGL is used for real-time result viewing. Besides, for high quality rendering, we use NVIDIA GVDB + OptiX Ray Tracing Engine.

### Overview

![Material point method overview](images/mpm_overview.png)

### References

##### Papers

- [Multi-species simulation of porous sand and water mixtures](https://www.math.ucla.edu/~jteran/papers/PGKFTJM17.pdf)
- [A material point method for snow simulation](https://www.math.ucla.edu/~jteran/papers/SSCTS13.pdf)

##### Other Implementations

- [Azmisov - snow](https://github.com/Azmisov/snow)
- [JAGJ10 - Snow](https://github.com/JAGJ10/Snow)
- [utilForever - SnowSimulation](https://github.com/utilForever/SnowSimulation)
