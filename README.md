MadNLP
========
MadNLP is a [noninear programming](https://en.wikipedia.org/wiki/Nonlinear_programming) (NLP) solver, purely implemented in [Julia](https://julialang.org/).

## Installation
```julia
pkg> add git@github.com:sshin23/MadNLP.git
```

## Build
**Automatic build is currently only supported for Linux and MacOS.** 

MadNLP is interfaced with non-julia sparse/dense linear solvers:
- [Umfpack](https://people.engr.tamu.edu/davis/suitesparse.html)
- [Mumps](http://mumps.enseeiht.fr/) 
- [MKL-Pardiso](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/sparse-solver-routines/intel-mkl-pardiso-parallel-direct-sparse-solver-interface.html) 
- [HSL solvers](http://www.hsl.rl.ac.uk/ipopt/) (optional) 
- [Pardiso](https://www.pardiso-project.org/) (optional) 
- [MKL-Lapack](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/lapack-routines.html)
- [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html)

All the dependencies except for HSL solvers and Pardiso are automatically obtained. To build MadNLP with HSL linear solvers (Ma27, Ma57, Ma77, Ma86, Ma97), the source codes need to be obtained by the user from <http://www.hsl.rl.ac.uk/ipopt/> under Coin-HSL Full (Stable). Then, the tarball `coinhsl-2015.06.23.tar.gz` should be placed at `deps/download`. To use Pardiso, the user needs to obtain the Paridso shared libraries from <https://www.pardiso-project.org/>, place the shared library file (e.g., `libpardiso600-GNU720-X86-64.so`) at `deps/download`, and place the license file in the home directory. After obtaining the files, run
```julia
pkg> build MadNLP
```
Other build dependencies include `gcc` and `gfortran`. Build can be customized by setting the following environment variables.
```julia
julia> ENV["MADNLP_CC"] = "/usr/local/bin/gcc-9"    # C compiler
julia> ENV["MADNLP_FC"] = "/usr/local/bin/gfortran" # Fortran compiler
julia> ENV["MADNLP_BLAS"] = "openblas"              # default is MKL
julia> ENV["MADNLP_ENALBE_OPENMP"] = false          # default is true
julia> ENV["MADNLP_OPTIMIZATION_FLAG"] = "-O2"      # default is -O3
```

## Bug reports and support
Please report issues and feature requests via the [Github issue tracker](https://github.com/sshin23/MadNLP/issues).

## Usage
MadNLP is interfaced with modeling packages: 
- [JuMP](https://github.com/jump-dev/JuMP.jl)
- [Plasmo](https://github.com/zavalab/Plasmo.jl)
- [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

### JuMP interface
```julia
using MadNLP, JuMP

model = Model(()->MadNLP.Optimizer(linear_solver="ma57",log_level="info",max_iter=100))
@variable(model, x, start = 0.0)
@variable(model, y, start = 0.0)
@NLobjective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2)

optimize!(model)

```

### Plasmo interface
```julia
using MadNLP, Plasmo

graph = OptiGraph()
@optinode(graph,n1)
@optinode(graph,n2)
@variable(n1,0 <= x <= 2)
@variable(n1,0 <= y <= 3)
@constraint(n1,x+y <= 4)
@objective(n1,Min,x)
@variable(n2,x)
@NLnodeconstraint(n2,exp(x) >= 2)
@linkconstraint(graph,n1[:x] == n2[:x])

MadNLP.optimize!(graph,ipopt;linear_solver="ma97",log_level="debug",max_iter=100)

```

### NLPModels interface
```julia
using MadNLP, CUTEst
model = CUTEstModel("PRIMALC1")
plamonlp(model,linear_solver="pardisomkl",log_level="warn",max_wall_time=3600)
```

## Options
Various algorithmic parameter that determines the solver behavior are passed as options. For interface-specific syntax for passing options, see the [Usage](https://github.com/sshin23/PlasmoNLP#usage).
- PlasmoNLP uses [Memento.jl](https://github.com/invenia/Memento.jl) for hierarchical logging. Option `log_level` and submodule log level options (e.g., `richardson_log_level`, `ma57_log_level`, ...) can be set as `"trace"`, `"debug"`, `"info"` (default), `"notice"`, `"warn"`, `"error"` (from less outputs to more outputs).
- `DefaultLinearSolver` is automatically set as `PlasmoNLP.Ma57` (`if` HSL solvers are available), `PlasmoNLP.Mumps` (`elseif` Mumps is available), or `PlasmoNLP.Umfpack` (`else`).
- `DefaultSubproblemSolver` is automatically set as `PlasmoNLP.Ma57` (`if` HSL solvers are available) or `PlasmoNLP.Umfpack` (`else`).
- `DefaultDenseSolver` is automatically set as `DenseLapack`, which uses [MKL LAPACK](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/lapack-routines.html) routines [^Bunch1977].
- Any option of type `Module` can be entered as strings. For examples, the following two are equivalent.
```julia
julia> model = Model(()->PlasmoNLP.Optimizer(linear_solver=PlasmoNLP.Ma57))
julia> model = Model(()->PlasmoNLP.Optimizer(linear_solver="ma57"))
```
- Any options of type `Bool` can be entered as `"yes"` or `"no"`.

### Interior Point Solver Options
- `linear_solver::Module = DefaultLinearSolver`:\
    Linear solver used for solving primal-dual system. Valid values are: `PlasmoNLP`.{`Umfpack`, `Mumps`, `PardisoMKL`, `Ma27`, `Ma57`, `Ma77`, `Ma86`, `Ma97`, `Pardiso`, `Schur`, `Schwarz`}.
- `iterative_solver::Module = Richardson `\
    Iterative solver used for iterative refinement. Valid values are: `PlasmoNLP`.{`Richardson`,`Krylov`}.
    - `Richardson` uses [Richardson iteration](https://en.wikipedia.org/wiki/Modified_Richardson_iteration)
    - `Krylov` uses [restarted Generalized Minimal Residual](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method) method implemented in [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl).
- `linear_system_scaler::Module = DummyModule`\
    Linear system scaling routine used for scaling primal-dual system. `DummyModule` does not scale the system. Valid values are {`DummyModule`,`PlasmoNLP.Mc19`}.
- `blas_num_threads::Int = Threads.nthreads()`\
    Number of threads used for BLAS routines. Valid range is ``[1,\infty)``.
- `disable_garbage_collector::Bool = false `\
    If `true`, Julia garbage collector is temporarily disabled while solving the problem, and then enabled back once the solution is complete.
- `rethrow_error::Bool = true `\
    f `false`, any internal error thrown by `PlasmoNLP` and interruption exception (triggered by the user via `^C`) is catched, and not rethrown. If an error is catched, the solver terminates with an error message.
- `log_level::String = "info"`\
    Log level for PlasmoNLP. The log level set here is propagated down to the submodules (e.g., `PlasmoNLP`.{`Richardson`, `Ma57`}). Valid values are: {`"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`}.
- `print_level::String = "trace"`\
    `stdout` print level. Any message with level less than `print_level` is not printed on `stdout`. Valid values are: {`"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`}.
- `output_file::String = ""`\
    If not `""`, the output log is teed to the file at the path specified in `output_file`. 
- `file_print_level::String = "trace"`\
    File print level; any message with level less than `file_print_level` is not printed on the file specified in `output_file`. Valid values are: {`"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`}.
- `tol::Float64 = 1e-8`\
    Termination tolerance. The solver terminates if the scaled primal, dual, complementary infeasibility is less than `tol`. Valid range is ``(0,\infty)``.
- `acceptable_tol::Float64 = 1e-6`\
    Acceptable tolerance. The solver terminates if the scaled primal, dual, complementary infeasibility is less than `acceptable_tol`, for `acceptable_iter` consecutive interior point iteration steps.
- `acceptable_iter::Int = 15`\
    Acceptable iteration tolerance. Valid rage is ``[1,\infty)``.
- `diverging_iterates_tol::Float64 = 1e20`\
    Diverging iteration tolerance. The solver terminates with exit symbol `:Diverging_Iterates` if the NLP error is greater than `diverging_iterates_tol`.
- `max_iter::Int = 3000`\
    Maximum number of interior point iterations. The solver terminates with exit symbol `:Maximum_Iterations_Exceeded` if the interior point iteration count exceeds `max_iter`.
- `max_wall_time::Float64 = 1e6`\
    Maximum wall time for interior point solver. The solver terminates with exit symbol `:Maximum_WallTime_Exceeded` if the total solver wall time exceeds `max_wall_time`.
- `fixed_variable_treatment::String = "relax_bounds"`\
    Valid values are: {`"relax_bounds"`,`"make_parameter"`}.    
- `jacobian_constant::Bool = false`\
    If `true`, constraint Jacobian is only evaluated once and reused.
- `hessian_constant::Bool = false`\
    If `true`, Lagrangian Hessian is only evaluated once and reused.
- `reduced_system::Bool = true`\
    If `true`, the primal-dual system is formulated as in [^Greif2014].
- `inertia_correction_method::String = "inertia_free"`\
    Valid values are: {`"inertia_based"`, `"inertia_free"`, `"inertia_ignored`"}.
    - `"ienrtia_based"` uses the strategy in [^Wächter2006],
    - `"inertia_free`" uses the strategy in [^Chiang2016]
    - `"inertia_ignored` simply ignores inertia information. 
- `s_max::Float64 = 100.`\
- `kappa_d::Float64 = 1e-5`\
- `constr_mult_init_max::Float64 = 1e3`\
- `bound_push::Float64 = 1e-2`\
- `bound_fac::Float64 = 1e-2`\
- `nlp_scaling_max_gradient::Float64 = 100.`\
- `inertia_free_tol::Float64 = 0.`\
- `min_hessian_perturbation::Float64 = 1e-20`\
- `first_hessian_perturbation::Float64 = 1e-4`\
- `max_hessian_perturbation::Float64 = 1e20`\
- `perturb_inc_fact_first::Float64 = 1e2`\
- `perturb_inc_fact::Float64 = 8.`\
- `perturb_dec_fact::Float64 = 1/3`\
- `jacobian_regularization_exponent::Float64 = 1/4`\
- `jacobian_regularization_value::Float64 = 1e-8`\
- `soft_resto_pderror_reduction_factor::Float64 = 0.9999`\
- `required_infeasibility_reduction::Float64 = 0.9`\
- `obj_max_inc::Float64 = 5.`\
- `kappha_soc::Float64 = 0.99`\
- `max_soc::Int = 4`\
- `alpha_min_frac::Float64 = 0.05`\
- `s_theta::Float64 = 1.1`\
- `s_phi::Float64 = 2.3`\
- `eta_phi::Float64 = 1e-4`\
- `kappa_soc::Float64 = 0.99`\
- `gamma_theta::Float64 = 1e-5`\
- `gamma_phi::Float64 = 1e-5`\
- `delta::Float64 = 1`\
- `kappa_sigma::Float64 = 1e10`\
- `barrier_tol_factor::Float64 = 10.`\
- `rho::Float64 = 1000.`\
- `mu_init::Float64 = 1e-1`\
- `mu_min::Float64 = 1e-9`\
- `mu_superlinear_decrease_power::Float64 = 1.5`\
- `tau_min::Float64 = 0.99`\
- `mu_linear_decrease_factor::Float64 = .2`\

### Linear Solver Options
Linear solver options are specific to the linear solver chosen at `linear_solver` option. Irrelevant options are ignored and a warning message is printed.
#### Ma27
- `ma27_pivtol::Float64 = 1e-8`\
- `ma27_pivtolmax::Float64 = 1e-4` \
- `ma27_liw_init_factor::Float64 = 5.` \
- `ma27_la_init_factor::Float64 = 5.` \
- `ma27_meminc_factor::Float64 = 2.` \
- `ma27_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Ma27`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### Ma57
- `ma57_pivtol::Float64 = 1e-8` \
- `ma57_pivtolmax::Float64 = 1e-4` \
- `ma57_pre_alloc::Float64 = 1.05` \
- `ma57_pivot_order::Int = 5` \
- `ma57_automatic_scaling::Bool = false` \
- `ma57_block_size::Int = 16` \
- `ma57_node_amalgamation::Int = 16` \
- `ma57_small_pivot_flag::Int = 0` \
- `ma57_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Ma57`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### Ma77
- `ma77_buffer_lpage::Int = 4096` \
- `ma77_buffer_npage::Int = 1600` \
- `ma77_file_size::Int = 2097152` \
- `ma77_maxstore::Int = 0` \
- `ma77_nemin::Int = 8` \
- `ma77_order::String = "metis"` \
- `ma77_print_level::Int = -1` \
- `ma77_small::Float64 = 1e-20` \
- `ma77_static::Float64 = 0.` \
- `ma77_u::Float64 = 1e-8` \
- `ma77_umax::Float64 = 1e-4` \
- `ma77_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Ma77`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### Ma86
- `ma86_num_threads::Int = 1` \
- `ma86_print_level::Float64 = -1` \
- `ma86_nemin::Int = 32` \
- `ma86_scaling::String = "none"` \
- `ma86_small::Float64 = 1e-20` \
- `ma86_static::Float64 = 0.` \
- `ma86_u::Float64 = 1e-8` \
- `ma86_umax::Float64 = 1e-4` \
- `ma86_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Ma86`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### Ma97
- `ma97_num_threads::Int = 1` \
- `ma97_print_level::Int = -1` \
- `ma97_nemin::Int = 8` \
- `ma97_order::String = "metis"` \
- `ma97_scaling::String = "none"` \
- `ma97_small::Float64 = 1e-20` \
- `ma97_u::Float64 = 1e-8` \
- `ma97_umax::Float64 = 1e-4` \
- `ma97_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Ma97`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### Mumps
- `mumps_dep_tol::Float64 = 0.` \
- `mumps_mem_percent::Int = 1000` \
- `mumps_permuting_scaling::Int = 7` \
- `mumps_pivot_order::Int = 7` \
- `mumps_pivtol::Float64 = 1e-6` \
- `mumps_pivtolmax::Float64 = .1` \
- `mumps_scaling::Int = 77` \
- `mumps_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Mumps`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### Umfpack
- `umfpack_pivtol::Float64 = 1e-4` \
- `umfpack_pivtolmax::Float64 = 1e-1` \
- `umfpack_sym_pivtol::Float64 = 1e-3` \
- `umfpack_block_size::Float64 = 16` \
- `umfpack_strategy::Float64 = 2.` \
- `umfpack_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Umfpack`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### Pardiso
- `pardiso_matching_strategy::String = "coplete+2x2"` \
- `pardiso_max_inner_refinement_steps::Int = 1` \
- `pardiso_msglvl::Int = 0` \
- `pardiso_order::Int = 2` \
- `pardiso_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Pardiso`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### PardisoMKL
- `pardisomkl_num_threads::Int = 1` \
- `pardisomkl_matching_strategy::String = "complete+2x2"` \
- `pardisomkl_max_iterative_refinement_steps::Int = 1` \
- `pardisomkl_msglvl::Int = 0` \
- `pardisomkl_order::Int = 2` \
- `pardisomkl_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.PardisoMKL`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### Schur
- `schur_subproblem_solver::Module = DefaultSubproblemSolver` \
   Linear solver used for solving subproblem. Valid values are: `PlasmoNLP`.{`Umfpack`, `Mumps`, `PardisoMKL`, `Ma27`, `Ma57`, `Ma77`, `Ma86`, `Ma97`, `Pardiso`, `Schur`, `Schwarz`}.
- `schur_dense_solver::Module = DefaultDenseSolver` \
   Lienar solver used for solving schur complement system
- `schur_custom_partition::Bool = false` \
   If `false`, Schur solver automatically detects the partition using `Metis`. If `true`, the partition information given in `schur_part` is used. `schur_num_parts` and `schur_part` should be properly set by the user. When using with `Plasmo`, `schur_num_parts` and `schur_part` are automatically set by the `Plasmo` interface. 
- `schur_num_parts::Int = 2` \
   Number of parts (excluding the parent node). Valid range is ``[1,\infty)``
- `schur_part::Vector{Int32} = Int32[]` \
   Custom partition information in a vector form. The parent node should be labeled as `0`. Only valid if `schur_custom_partition` is `true`.
- `schur_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Schur`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### Schwarz
- `schwarz_subproblem_solver::Module = DefaultSubproblemSolver` \
   Linear solver used for solving subproblem. Valid values are: `PlasmoNLP`.{`Umfpack`, `Mumps`, `PardisoMKL`, `Ma27`, `Ma57`, `Ma77`, `Ma86`, `Ma97`, `Pardiso`, `Schur`, `Schwarz`}.
- `schwarz_custom_partition::Bool = false` \
    If `false`, Schwarz solver  automatically detects the partition using `Metis`. If `true`, the partition information given in `schur_part` is used. `schur_num_parts` and `schur_part` should be properly set by the user. When using with `Plasmo`, `schur_num_parts` and `schur_part` are automatically set by the `Plasmo` interface. 
- `schwarz_num_parts::Int = 2` \
    Number of parts. Valid range is ``[1,\infty)``
- `schwarz_part::Vector{Int32} = Int32[]` \
    Custom partition information in a vector form. Only valid if `schwar_custom_partition` is `true`.
- `schwarz_num_parts_upper::Int = 0` \
    Number of parts in upper level partition. If `schwarz_num_parts_upper!=0`, a bilevel partitioniong scheme is used. Valid range is ``[1,\infty)``
- `schwarz_part_upper::Vector{Int32} = Int32[]` \
    Custom partition for the upper level partition. 
- `schwarz_fully_improve_subproblem_solver::Bool = true` \
    If `true`, the subproblem solvers are fully improved when the linear solver is initialized.
- `schwarz_max_expand_factor::Int = 4` \
    The size of overlap is fully saturated when the `improve!` is called `schwarz_max_expand_factor-1` times. Valid range is ``[2,\infty)``.
- `schwarz_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Schwarz`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

### Iterator Options
#### Richardson
- `richardson_max_iter::Int = 10` \
    Maximum number of Richardson iteration steps. Valid range is ``[1,\infty)``.
- `richardson_tol::Float64 = 1e-10` \
    Convergence tolerance of Richardson iteration. Valid range is ``(0,\infty)``.
- `richardson_acceptable_tol::Float64 = 1e-5` \
    Acceptable convergence tolerance of Richardson iteration. If the Richardson iteration counter exceeds `richardson_max_iter` without satisfying the convergence criteria set with `richardson_tol`, the Richardson solver checks whether the acceptable convergence criteria set with `richardson_acceptable_tol` is satisfied; if the acceptable convergence criteria is satisfied, the computed step is used; otherwise, the augmented system is treated to be singular. Valid range is ``(0,\infty)``.
- `richardson_log_level::String = ""` \
   Log level for submodule `PlasmoNLP.Richardson`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

#### Krylov
- `krylov_max_iter::Int = 10` \
    Maximum number of Krylov iteration steps. Valid range is ``[1,\infty)``.
- `krylov_tol::Float64 = 1e-10` \
    Convergence tolerance of Krylov iteration. Valid range is ``(0,\infty)``.
- `krylov_acceptable_tol::Float64 = 1e-5` \
    Acceptable convergence tolerance of Krylov iteration. If the Krylov iteration counter exceeds `krylov_max_iter` without satisfying the convergence criteria set with `krylov_tol`, the Krylov solver checks whether the acceptable convergence criteria set with `krylov_acceptable_tol` is satisfied; if the acceptable convergence criteria is satisfied, the computed step is used; otherwise, the augmented system is treated to be singular. Valid range is ``(0,\infty)``.
- `krylov_restart::Int = 5` \
    Maximum Krylov iteration before restarting. Valid range is ``[1,\infty)``.
- `krylov_log_level::String = ""` \
    Log level for submodule `PlasmoNLP.Krylov`. Valid values are: `"trace"`, `"debug"`, `"info"`, `"notice"`, `"warn"`, `"error"`.

### Reference
[^Bunch1977]: J R Bunch and L Kaufman, Some stable methods for calculating inertia and solving symmetric linear systems, Mathematics of Computation 31:137 (1977), 163-179.

[^Greif2014]: Greif, Chen, Erin Moulding, and Dominique Orban. "Bounds on eigenvalues of matrices arising from interior-point methods." SIAM Journal on Optimization 24.1 (2014): 49-83.

[^Wächter2006]: Wächter, Andreas, and Lorenz T. Biegler. "On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming." Mathematical programming 106.1 (2006): 25-57.

[^Chiang2016]: Chiang, Nai-Yuan, and Victor M. Zavala. "An inertia-free filter line-search algorithm for large-scale nonlinear programming." Computational Optimization and Applications 64.2 (2016): 327-354.

