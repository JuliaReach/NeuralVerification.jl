# This file is for different constraints

abstract type AbstractLinearProgram end
struct StandardLP            <: AbstractLinearProgram end
struct LinearRelaxedLP       <: AbstractLinearProgram end
struct TriangularRelaxedLP   <: AbstractLinearProgram end
struct BoundedMixedIntegerLP <: AbstractLinearProgram end
struct SlackLP <: AbstractLinearProgram
    slack::Vector{Vector{VariableRef}}
end
SlackLP() = SlackLP([])
struct MixedIntegerLP <: AbstractLinearProgram
    m::Float64
end

# Any encoding passes through here first:
function encode_network!(model::Model,
                         network::Network,
                         zs::Vector{Vector{VariableRef}},
                         δs::Vector,
                         encoding::AbstractLinearProgram)

    for (i, layer) in enumerate(network.layers)
        encode_layer!(encoding, model, layer, zs[i], zs[i+1], δs[i])
    end
    return encoding # only matters for SlackLP
end

# TODO: find a way to eliminate the two methods below.
# i.e. make BoundedMixedIntegerLP(bounds) and TriangularRelaxedLP(bounds) or something
function encode_network!(model::Model,
                         network::Network,
                         zs::Vector{Vector{VariableRef}},
                         δs::Vector,
                         bounds::Vector{Hyperrectangle},
                         encoding::AbstractLinearProgram)

    for (i, layer) in enumerate(network.layers)
        encode_layer!(encoding, model, layer, zs[i], zs[i+1], δs[i], bounds[i])
    end
    return encoding # only matters for SlackLP
end

function encode_network!(model::Model,
                         network::Network,
                         zs::Vector{Vector{VariableRef}},
                         bounds::Vector{Hyperrectangle},
                         encoding::AbstractLinearProgram)

    for (i, layer) in enumerate(network.layers)
        encode_layer!(encoding, model, layer, zs[i], zs[i+1], bounds[i])
    end
    return encoding # only matters for SlackLP
end


# For an Id Layer, any encoding type defaults to this:
function encode_layer!(::AbstractLinearProgram,
                       model::Model,
                       layer::Layer{Id},
                       zᵢ::Vector{VariableRef},
                       zᵢ₊₁::Vector{VariableRef},
                       args...)
    @constraint(model, zᵢ₊₁ .== affine_map(layer, zᵢ))
end

# SlackLP is slightly different, because we need to keep track of the slack variables
function encode_layer!(SLP::SlackLP,
                       model::Model,
                       layer::Layer{Id},
                       zᵢ::Array{VariableRef,1},
                       zᵢ₊₁::Array{VariableRef,1},
                       δ...)

    encode_layer!(StandardLP(), model, layer, zᵢ, zᵢ₊₁)
    # We need identity layer slack variables so that the algorithm doesn't
    # "get confused", but they are set to 0 because they're not relevant
    slack_vars = @variable(model, [1:n_nodes(layer)])
    @constraint(model, slack_vars .== 0.0)
    push!(SLP.slack, slack_vars)
    return nothing
end

# alternative signature:
# encode_layer!(encoding, model::Model, current_layer::VarLayer{ReLU},  next_layer::VarLayer)
function encode_layer!(::StandardLP,
                       model::Model,
                       layer::Layer{ReLU},
                       zᵢ::Vector{VariableRef},
                       zᵢ₊₁::Vector{VariableRef},
                       δᵢ₊₁::Vector{Bool})

    # The jth ReLU is forced to be active or inactive,
    # depending on the activation pattern given by δᵢ₊₁.
    # δᵢⱼ == true denotes ẑ >=0 (i.e. an *inactive* ReLU)
    ẑ = affine_map(layer, zᵢ)
    for j in 1:length(layer.bias)
        if δᵢ₊₁[j]
            @constraint(model, ẑ[j] >= 0.0)
            @constraint(model, zᵢ₊₁[j] == ẑ[j])
        else
            @constraint(model, ẑ[j] <= 0.0)
            @constraint(model, zᵢ₊₁[j] == 0.0)
        end
    end
end

function encode_layer!(SLP::SlackLP,
                       model::Model,
                       layer::Layer{ReLU},
                       zᵢ::Vector{VariableRef},
                       zᵢ₊₁::Vector{VariableRef},
                       δᵢ₊₁::Vector{Bool})

    ẑ = affine_map(layer, zᵢ)
    slack_vars = @variable(model, [1:length(layer.bias)])
    for j in 1:length(layer.bias)
        if δᵢ₊₁[j]
            @constraint(model, zᵢ₊₁[j] == ẑ[j] + slack_vars[j])
            @constraint(model, ẑ[j] + slack_vars[j] >= 0.0)
        else
            @constraint(model, zᵢ₊₁[j] == slack_vars[j])
            @constraint(model, 0.0 >= ẑ[j] - slack_vars[j])
        end
    end
    push!(SLP.slack, slack_vars)
    return nothing
end

function encode_layer!(::LinearRelaxedLP,
                       model::Model,
                       layer::Layer{ReLU},
                       zᵢ::Vector{VariableRef},
                       zᵢ₊₁::Vector{VariableRef},
                       δᵢ₊₁::Vector{Bool})

    ẑ = affine_map(layer, zᵢ)
    for j in 1:length(layer.bias)
        if δᵢ₊₁[j]
            @constraint(model, zᵢ₊₁[j] == ẑ[j])
        else
            @constraint(model, zᵢ₊₁[j] == 0.0)
        end
    end
end


function encode_layer!(::TriangularRelaxedLP,
                       model::Model,
                       layer::Layer{ReLU},
                       zᵢ::Vector{VariableRef},
                       zᵢ₊₁::Vector{VariableRef},
                       bounds::Hyperrectangle)

    ẑ = affine_map(layer, zᵢ)
    ẑ_bound = approximate_affine_map(layer, bounds)
    l̂, û = low(ẑ_bound), high(ẑ_bound)
    for j in 1:length(layer.bias)
        if l̂[j] > 0.0
            @constraint(model, zᵢ₊₁[j] == ẑ[j])
        elseif û[j] < 0.0
            @constraint(model, zᵢ₊₁[j] == 0.0)
        else
            @constraints(model, begin
                                    zᵢ₊₁[j] >= ẑ[j]
                                    zᵢ₊₁[j] <= û[j] / (û[j] - l̂[j]) * (ẑ[j] - l̂[j])
                                    zᵢ₊₁[j] >= 0.0
                                end)
        end
    end
end

function encode_layer!(MIP::MixedIntegerLP,
                       model::Model,
                       layer::Layer{ReLU},
                       zᵢ::Vector{VariableRef},
                       zᵢ₊₁::Vector{VariableRef},
                       δᵢ₊₁::Vector{VariableRef})
    m = MIP.m

    ẑ = affine_map(layer, zᵢ)
    for j in 1:length(layer.bias)
        @constraints(model, begin
                                zᵢ₊₁[j] >= ẑ[j]
                                zᵢ₊₁[j] >= 0.0
                                zᵢ₊₁[j] <= ẑ[j] + m * δᵢ₊₁[j]
                                zᵢ₊₁[j] <= m - m * δᵢ₊₁[j]
                            end)
    end
end

function encode_layer!(::BoundedMixedIntegerLP,
                       model::Model,
                       layer::Layer{ReLU},
                       zᵢ::Vector{VariableRef},
                       zᵢ₊₁::Vector{VariableRef},
                       δᵢ₊₁::Vector,
                       bounds::Hyperrectangle)

    ẑ = affine_map(layer, zᵢ)
    ẑ_bound = approximate_affine_map(layer, bounds)
    l̂, û = low(ẑ_bound), high(ẑ_bound)
    for j in 1:length(layer.bias) # For evey node
        if l̂[j] >= 0.0
            @constraint(model, zᵢ₊₁[j] == ẑ[j])
        elseif û[j] <= 0.0
            @constraint(model, zᵢ₊₁[j] == 0.0)
        else
            @constraints(model, begin
                                    zᵢ₊₁[j] >= ẑ[j]
                                    zᵢ₊₁[j] >= 0.0
                                    zᵢ₊₁[j] <= û[j] * δᵢ₊₁[j]
                                    zᵢ₊₁[j] <= ẑ[j] - l̂[j] * (1 - δᵢ₊₁[j])
                                end)
        end
    end
end


#=
Add input/output constraints to model
=#
function add_complementary_set_constraint!(model::Model, output::HPolytope, z::Vector{VariableRef})
    out_A, out_b = tosimplehrep(output)
    # Needs to take the complementary of output constraint
    n = length(constraints_list(output))
    if n == 1
        # Here the output constraint is a half space
        halfspace = first(constraints_list(output))
        add_complementary_set_constraint!(model, halfspace, z)
    else
        error("Non-convex constraints are not supported. Please make sure that the
            output set is a HalfSpace (or an HPolytope with a single constraint) so that the
            complement of the output is convex. Got $n constraints.")
    end
    return nothing
end

function add_complementary_set_constraint!(m::Model, H::HalfSpace, z::Vector{VariableRef})
    a, b = tosimplehrep(H)
    @constraint(m, a * z .>= b)
    return nothing
end
function add_complementary_set_constraint!(m::Model, PC::PolytopeComplement, z::Vector{VariableRef})
    add_set_constraint!(m, PC.P, z)
    return nothing
end

function add_set_constraint!(m::Model, set::Union{HPolytope, HalfSpace}, z::Vector{VariableRef})
    A, b = tosimplehrep(set)
    @constraint(m, A * z .<= b)
    return nothing
end

function add_set_constraint!(m::Model, set::Hyperrectangle, z::Vector{VariableRef})
    @constraint(m, z .<= high(set))
    @constraint(m, z .>= low(set))
    return nothing
end

function add_set_constraint!(m::Model, PC::PolytopeComplement, z::Vector{VariableRef})
    add_complementary_set_constraint!(m, PC.P, z)
    return nothing
end
