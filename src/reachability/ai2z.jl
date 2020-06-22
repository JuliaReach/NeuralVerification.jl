"""
    Ai2z <: AbstractSolver

Ai2 performs over-approximated reachability analysis to compute the over-approximated output reachable set for a network.

# Problem requirement
1. Network: any depth, ReLU activation (more activations to be supported in the future)
2. Input: Zonotope
3. Output: Zonotope

# Return
`ReachabilityResult`

# Method
Reachability analysis using split and join using Zonotopes as proposed on [1].

# Property
Sound but not complete.

# Reference
T. Gehr, M. Mirman, D. Drashsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev,
"Ai2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation,"
in *2018 IEEE Symposium on Security and Privacy (SP)*, 2018.
"""
struct Ai2z <: AbstractSolver end

function solve(solver::Ai2z, problem::Problem)
    reach = forward_network(solver, problem.network, problem.input)
    return check_inclusion(reach, problem.output)
end

forward_layer(solver::Ai2z, layer::Layer, inputs::Vector{<:Zonotope}) = forward_layer.(solver, layer, inputs)

function forward_layer(solver::Ai2z, layer::Layer, input::Zonotope)
    outlinear = affine_map(layer, input)
    relued_subsets = forward_partition(layer.activation, outlinear)
    return relued_subsets
end
