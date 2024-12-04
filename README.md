# GR3D.jl
Numerical Relativity simulations using `ParallelStencil.jl` and embedded boundary finite differencing operators to allow for arbitrary 3D domains.

This Julia program was used to produce the results published here: https://arxiv.org/abs/2409.17970

This applies several novel frameworks
  - Demonstrates embedded boundary SBP finite differencing stencils
  - Demonstrates a novel SBP based numerical relativity formulation based on generalized harmonic
  - Implements a novel boundary condition framework based on first derivatives of the metric

The setup is a cubical domain with a spherical hole cut out in the center, inside which is placed a black hole.
Gravitational waves are then injected during the evolution, perturbing the black hole.

To run this, make sure you have started Julia with the desired amount of threads and have installed all necessary packages. **Use Tensorial.jl version previous to v0.17.0 as something has changed to trash performance. Working on figuring out why this happens.**

Then run `include("src/main.jl")`, and finally run `GR3D.main()`. 

Check out `src/main.jl` for various settings involving spacetime domain size, inner boundary radius, etc.
Check out `src/initial_conditions.jl` for setting the mass and spin of the black hole
Check out `src/boundaries.jl` for different boundary conditions one might consder at the inner boundary


