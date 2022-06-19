const dummy_function = ()->nothing

num_linkconstraints(modeledge::OptiEdge) = length(modeledge.linkconstraints)
function _caching_optimizer(modelnode::OptiNode)
    if isa(modelnode.model.moi_backend,MOIU.CachingOptimizer)
        return modelnode.model.moi_backend
    else
        return modelnode.model.moi_backend.optimizer
    end
end
moi_optimizer(modelnode::OptiNode) = _caching_optimizer(modelnode).optimizer.model
_caching_optimizer(model::Any) = model.moi_backend
function set_g_link!(linkedge::OptiEdge,l,gl,gu)
    cnt = 1
    for (ind,linkcon) in linkedge.linkconstraints
        l[cnt] = 0. # need to implement dual start later
        if linkcon.set isa MOI.EqualTo
            gl[cnt] = linkcon.set.value
            gu[cnt] = linkcon.set.value
        elseif linkcon.set isa MOI.GreaterThan
            gl[cnt] = linkcon.set.lower
            gu[cnt] = Inf
        elseif linkcon.set isa MOI.LessThan
            gl[cnt] = -Inf
            gu[cnt] = linkcon.set.upper
        else
            gl[cnt] = linkcon.set.lower
            gu[cnt] = linkcon.set.upper
        end
        cnt += 1
    end
end

function hessian_lagrangian_structure(graph::OptiGraph,I,J,ninds,nnzs_hess_inds,modelnodes)
    @blas_safe_threads for k=1:length(modelnodes)
        isempty(nnzs_hess_inds[k]) && continue
        offset = ninds[k][1]-1
        II = view(I,nnzs_hess_inds[k])
        JJ = view(J,nnzs_hess_inds[k])
        hessian_lagrangian_structure(
            moi_optimizer(modelnodes[k]),II,JJ)
        II.+= offset
        JJ.+= offset
    end
end

function jacobian_structure(linkedge::OptiEdge,I,J,ninds,x_index_map,g_index_map)
    offset=1
    for linkcon in link_constraints(linkedge)
        offset += jacobian_structure(linkcon,I,J,ninds,x_index_map,g_index_map,offset)
    end
end
function jacobian_structure(linkcon,I,J,ninds,x_index_map,g_index_map,offset)
    cnt = 0
    for var in get_vars(linkcon)
        I[offset+cnt] = g_index_map[linkcon]
        J[offset+cnt] = x_index_map[var]
        cnt += 1
    end
    return cnt
end

function jacobian_structure(
    graph::OptiGraph,I,J,ninds,minds,pinds,nnzs_jac_inds,nnzs_link_jac_inds,
    x_index_map,g_index_map,modelnodes,linkedges)

    @blas_safe_threads for k=1:length(modelnodes)
        isempty(nnzs_jac_inds[k]) && continue
        offset_i = minds[k][1]-1
        offset_j = ninds[k][1]-1
        II = view(I,nnzs_jac_inds[k])
        JJ = view(J,nnzs_jac_inds[k])
        jacobian_structure(
            moi_optimizer(modelnodes[k]),II,JJ)
        II.+= offset_i
        JJ.+= offset_j
    end

    @blas_safe_threads for q=1:length(linkedges)
        isempty(nnzs_link_jac_inds[q]) && continue
        II = view(I,nnzs_link_jac_inds[q])
        JJ = view(J,nnzs_link_jac_inds[q])
        jacobian_structure(
            linkedges[q],II,JJ,ninds,x_index_map,g_index_map)
    end
end
function eval_objective(graph::OptiGraph,x,ninds,x_index_map,modelnodes)
    obj = Threads.Atomic{Float64}(0.)
    @blas_safe_threads for k=1:length(modelnodes)
         Threads.atomic_add!(obj,eval_objective(
             moi_optimizer(modelnodes[k]),view(x,ninds[k])))
    end
    return obj.value + eval_function(graph.objective_function,x,ninds,x_index_map)
end
function eval_objective_gradient(graph::OptiGraph,f,x,ninds,modelnodes)
    @blas_safe_threads for k=1:length(modelnodes)
        eval_objective_gradient(moi_optimizer(modelnodes[k]),view(f,ninds[k]),view(x,ninds[k]))
    end
end

function eval_function(aff::GenericAffExpr,x,ninds,x_index_map)
    function_value = aff.constant
    for (var,coef) in aff.terms
        function_value += coef*x[x_index_map[var]]
    end
    return function_value
end
function eval_constraint(linkedge::OptiEdge,c,x,ninds,x_index_map)
    cnt = 1
    for linkcon in link_constraints(linkedge)
        c[cnt] = eval_function(get_func(linkcon),x,ninds,x_index_map)
        cnt += 1
    end
end
get_func(linkcon) = linkcon.func
function eval_constraint(graph::OptiGraph,c,x,ninds,minds,pinds,x_index_map,modelnodes,linkedges)
    @blas_safe_threads for k=1:length(modelnodes)
        eval_constraint(moi_optimizer(modelnodes[k]),view(c,minds[k]),view(x,ninds[k]))
    end
    @blas_safe_threads for q=1:length(linkedges)
        eval_constraint(linkedges[q],view(c,pinds[q]),x,ninds,x_index_map)
    end
end

function eval_hessian_lagrangian(graph::OptiGraph,hess,x,sig,l,
                                 ninds,minds,nnzs_hess_inds,modelnodes)
    @blas_safe_threads for k=1:length(modelnodes)
        isempty(nnzs_hess_inds) && continue
        eval_hessian_lagrangian(moi_optimizer(modelnodes[k]),
                                view(hess,nnzs_hess_inds[k]),view(x,ninds[k]),sig,
                                view(l,minds[k]))
    end
end

function eval_constraint_jacobian(linkedge::OptiEdge,jac,x)
    offset=0
    for linkcon in link_constraints(linkedge)
        offset+=eval_constraint_jacobian(linkcon,jac,offset)
    end
end
function eval_constraint_jacobian(linkcon,jac,offset)
    cnt = 0
    for coef in get_coeffs(linkcon)
        cnt += 1
        jac[offset+cnt] = coef
    end
    return cnt
end
get_vars(linkcon) = keys(linkcon.func.terms)
get_coeffs(linkcon) = values(linkcon.func.terms)

function eval_constraint_jacobian(graph::OptiGraph,jac,x,
                                  ninds,minds,nnzs_jac_inds,nnzs_link_jac_inds,modelnodes,linkedges)
    @blas_safe_threads for k=1:length(modelnodes)
        eval_constraint_jacobian(
            moi_optimizer(modelnodes[k]),view(jac,nnzs_jac_inds[k]),view(x,ninds[k]))
    end
    @blas_safe_threads for q=1:length(linkedges)
        eval_constraint_jacobian(linkedges[q],view(jac,nnzs_link_jac_inds[q]),x)
    end
end
get_nnz_link_jac(linkedge::OptiEdge) = sum(
    length(linkcon.func.terms) for (ind,linkcon) in linkedge.linkconstraints)


struct GraphModel{T} <: AbstractNLPModel{T,Vector{T}}
    ninds::Vector{UnitRange{Int}}
    minds::Vector{UnitRange{Int}}
    pinds::Vector{UnitRange{Int}}
    nnzs_jac_inds::Vector{UnitRange{Int}}
    nnzs_hess_inds::Vector{UnitRange{Int}}
    nnzs_link_jac_inds::Vector{UnitRange{Int}}

    x_index_map::Dict #
    g_index_map::Dict #

    modelnodes::Vector{OptiNode}
    linkedges::Vector{OptiEdge}

    meta::NLPModelMeta{T, Vector{T}}
    graph::OptiGraph
    counters::NLPModelsCounters
    ext::Dict{Symbol,Any}
end

obj(nlp::GraphModel, x::AbstractVector) =  eval_objective(nlp.graph,x,nlp.ninds,nlp.x_index_map,nlp.modelnodes)
grad!(nlp::GraphModel, x::AbstractVector, f::AbstractVector) =eval_objective_gradient(nlp.graph,f,x,nlp.ninds,nlp.modelnodes)
cons!(nlp::GraphModel, x::AbstractVector, c::AbstractVector) = eval_constraint(
    nlp.graph,c,x,nlp.ninds,nlp.minds,nlp.pinds,nlp.x_index_map,nlp.modelnodes,nlp.linkedges)
function hess_coord!(nlp::GraphModel,x::AbstractVector,l::AbstractVector,hess::AbstractVector;obj_weight=1.)
    eval_hessian_lagrangian(
        nlp.graph,hess,x,obj_weight,l,nlp.ninds,nlp.minds,
        nlp.nnzs_hess_inds,nlp.modelnodes,
    )
end
function jac_coord!(nlp::GraphModel,x::AbstractVector,jac::AbstractVector)
    eval_constraint_jacobian(
        nlp.graph,jac,x,nlp.ninds,nlp.minds,nlp.nnzs_jac_inds,nlp.nnzs_link_jac_inds,
        nlp.modelnodes,nlp.linkedges,
    )
end
function hess_structure!(nlp::GraphModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    hessian_lagrangian_structure(
        nlp.graph,I,J,nlp.ninds,nlp.nnzs_hess_inds,nlp.modelnodes,
    )
end
function jac_structure!(nlp::GraphModel, I::AbstractVector{T}, J::AbstractVector{T}) where T
    jacobian_structure(
        nlp.graph,I,J,nlp.ninds,nlp.minds,nlp.pinds,nlp.nnzs_jac_inds,nlp.nnzs_link_jac_inds,
        nlp.x_index_map,nlp.g_index_map,nlp.modelnodes,nlp.linkedges,
    )
end


"""
    GraphModel(graph::OptiGraph; use_subgraphs=False)
"""
function GraphModel(graph::OptiGraph)

    modelnodes = all_nodes(graph)
    linkedges = all_edges(graph)

    for modelnode in modelnodes
        num_variables(modelnode) == 0 && error("Empty node exist! Delete the empty nodes.")
    end

    @blas_safe_threads for k=1:length(modelnodes)
        set_optimizer(modelnodes[k],Optimizer)
        if modelnodes[k].model.nlp_data !== nothing
            MOI.set(modelnodes[k].model, MOI.NLPBlock(),
                    _create_nlp_block_data(modelnodes[k].model))
            empty!(modelnodes[k].model.nlp_data.nlconstr_duals)
        end
        MOIU.attach_optimizer(modelnodes[k].model)
        MOI.initialize(moi_optimizer(modelnodes[k]).nlp_data.evaluator,[:Grad,:Hess,:Jac])
    end

    K = length(modelnodes)
    ns= [num_variables(modelnode) for modelnode in modelnodes]
    n = sum(ns)
    ns_cumsum = cumsum(ns)
    ms= [num_constraints(modelnode) for modelnode in modelnodes]
    ms_cumsum = cumsum(ms)
    m = sum(ms)

    nnzs_hess = [get_nnz_hess(moi_optimizer(modelnode)) for modelnode in modelnodes]
    nnzs_hess_cumsum = cumsum(nnzs_hess)
    nnz_hess = sum(nnzs_hess)

    nnzs_jac = [get_nnz_jac(moi_optimizer(modelnode)) for modelnode in modelnodes]
    nnzs_jac_cumsum = cumsum(nnzs_jac)
    nnz_jac = sum(nnzs_jac)

    nnzs_link_jac = [get_nnz_link_jac(linkedge) for linkedge in linkedges]
    nnzs_link_jac_cumsum = cumsum(nnzs_link_jac)
    nnz_link_jac = isempty(nnzs_link_jac) ? 0 : sum(nnzs_link_jac)

    ninds = [(i==1 ? 0 : ns_cumsum[i-1])+1:ns_cumsum[i] for i=1:K]
    minds = [(i==1 ? 0 : ms_cumsum[i-1])+1:ms_cumsum[i] for i=1:K]
    nnzs_hess_inds = [(i==1 ? 0 : nnzs_hess_cumsum[i-1])+1:nnzs_hess_cumsum[i] for i=1:K]
    nnzs_jac_inds = [(i==1 ? 0 : nnzs_jac_cumsum[i-1])+1:nnzs_jac_cumsum[i] for i=1:K]

    Q = length(linkedges)
    ps= [num_linkconstraints(modeledge) for modeledge in linkedges]
    ps_cumsum =  cumsum(ps)
    p = sum(ps)
    pinds = [(i==1 ? m : m+ps_cumsum[i-1])+1:m+ps_cumsum[i] for i=1:Q]
    nnzs_link_jac_inds =
        [(i==1 ? nnz_jac : nnz_jac+nnzs_link_jac_cumsum[i-1])+1: nnz_jac + nnzs_link_jac_cumsum[i] for i=1:Q]

    x =Vector{Float64}(undef,n)
    xl=Vector{Float64}(undef,n)
    xu=Vector{Float64}(undef,n)

    l =Vector{Float64}(undef,m+p)
    gl=Vector{Float64}(undef,m+p)
    gu=Vector{Float64}(undef,m+p)

    @blas_safe_threads for k=1:K
        set_x!(moi_optimizer(modelnodes[k]),view(x,ninds[k]),view(xl,ninds[k]),view(xu,ninds[k]))
        set_g!(moi_optimizer(modelnodes[k]),view(l,minds[k]),view(gl,minds[k]),view(gu,minds[k]))
    end

    @blas_safe_threads for q=1:Q
        set_g_link!(linkedges[q],view(l,pinds[q]),view(gl,pinds[q]),view(gu,pinds[q]))
    end

    modelmap=Dict(modelnodes[k].model=> k for k=1:K)
    x_index_map = Dict(
        # var=>ninds[modelmap[var.model]][backend(var.model).optimizer.model_to_optimizer_map[var.index].value]
        var=>ninds[modelmap[var.model]][_caching_optimizer(getnode(var)).model_to_optimizer_map[var.index].value]
        for modelnode in modelnodes for var in all_variables(modelnode))
    cnt = 0
    g_index_map = Dict(con=> m + (cnt+=1) for linkedge in linkedges for (ind,con) in linkedge.linkconstraints)

    jac_constant = true
    hess_constant = true
    for node in modelnodes
        j,h = is_jac_hess_constant(moi_optimizer(node))
        jac_constant = jac_constant & j
        hess_constant = hess_constant & h
    end

    ext = Dict{Symbol,Any}(:n=>n,:m=>m,:p=>p,:ninds=>ninds,:minds=>minds,:pinds=>pinds,
                           :linkedges=>linkedges,:jac_constant=>jac_constant,:hess_constant=>hess_constant)

    return GraphModel(
        ninds,minds,pinds,nnzs_jac_inds,nnzs_hess_inds,nnzs_link_jac_inds,
        x_index_map,g_index_map,modelnodes,linkedges,
        NLPModelMeta(
            n,
            ncon = m+p,
            x0 = x,
            lvar = xl,
            uvar = xu,
            y0 = l,
            lcon = gl,
            ucon = gu,
            nnzj = nnz_jac + nnz_link_jac,
            nnzh = nnz_hess,
            minimize = true # graph.objective_sense == MOI.MIN_SENSE
        ),
        graph,NLPModelsCounters(),
        ext
    )

end

function get_part(graph::OptiGraph,nlp::GraphModel,partition_type::Symbol)
    if partition_type == :nodes
        return get_part_nodes(graph,nlp)
    elseif partition_type == :subgraphs
        return get_part_subgraphs(graph,nlp)
    else
        error("Invalid option passed as partition type. Must specific :nodes or :subgraphs")
    end
end

#get partition of the GraphModel using nodes.
#This ultimately produces a partition vector of variables and constraints
function get_part_nodes(graph::OptiGraph,nlp::GraphModel)
    n = nlp.ext[:n] #num variables
    m = nlp.ext[:m] #num constraints
    p = nlp.ext[:p] #num link constraints

    ninds = nlp.ninds #variable partitions [1:n_vars]
    minds = nlp.minds #constraint partitions [1:n_cons]
    pinds = nlp.pinds #linking constraints [n_cons+1 : n_cons + n_links]

    #inequality constraint indices
    ind_ineq = findall(get_lcon(nlp).!=get_ucon(nlp))
    l = length(ind_ineq)

    #part vector elements are: n_variables + n_constraints + n_link_constraints
    part = Vector{Int}(undef,n+m+l+p)

    #assign variables to partitions
    for k=1:length(ninds)
        part[ninds[k]].=k
    end

    #assign constraints to partitions
    for k=1:length(minds)
        part[minds[k].+n.+l].=k
    end

    #assign link constraints to partitions based on attached node
    #linkedges should be highest level edges
    cnt = 0
    for linkedge in nlp.ext[:linkedges]
        for (ind,con) in linkedge.linkconstraints
            cnt+=1
            attached_node_idx = graph.node_idx_map[con.attached_node]
            part[n+l+m+cnt] = attached_node_idx != nothing ? attached_node_idx : error("All the link constraints need to be attached to a node")
        end
    end

    #assign inequality constraints
    cnt = 0
    for q in ind_ineq
        cnt+=1
        part[n+cnt] = part[n+l+q]
    end

    return part
end

function get_part_subgraphs(graph::OptiGraph,nlp::GraphModel)

    n = nlp.ext[:n] #variables
    m = nlp.ext[:m] #constraints
    p = nlp.ext[:p] #link constraints

    #TODO: re-assign based on subgraphs
    ninds = nlp.ninds
    minds = nlp.minds
    pinds = nlp.pinds

    #define partition based on subgraph structure

    #inequality constraint indices
    ind_ineq = findall(get_lcon(nlp).!=get_ucon(nlp))
    l = length(ind_ineq)

    #part vector elements are: n_variables + n_constraints + n_link_constraints
    part = Vector{Int}(undef,n+m+l+p)
    #assign variables to partitions
    for k=1:length(ninds)
        part[ninds[k]].=k
    end

    #assign constraints to partitions
    for k=1:length(minds)
        part[minds[k].+n.+l].=k
    end
    cnt = 0

    #assign link constraints to partitions
    #linkedges should be highest level edges
    for linkedge in nlp.ext[:linkedges]
        for (ind,con) in linkedge.linkconstraints
            cnt+=1
            attached_node_idx = graph.node_idx_map[con.attached_node]
            part[n+l+m+cnt] = attached_node_idx != nothing ? attached_node_idx : error("All the link constraints need to be attached to a node")
        end
    end

    #assign inequality constraints
    cnt = 0
    for q in ind_ineq
        cnt+=1
        part[n+cnt] = part[n+l+q]
    end

    return part
end

function _get_option(option::Symbol,option_dict::Dict{Symbol,Any},kwargs...)
    if haskey(kwargs,option)
        return kwargs[option]
    elseif haskey(option_dict,option)
        return option_dict[option]
    else
        return nothing
    end
end

function _set_schur_options!(graph::OptiGraph,nlp::GraphModel,option_dict::Dict,partition::Symbol)
    part = get_part(graph,nlp,partition)
    K = length(unique(part))
    # part[part.>K].=0 #NOTE: this line might not be needed
    option_dict[:schur_part] = part
    option_dict[:schur_num_parts] = K
    return nothing
end

function _set_schwarz_options!(graph::OptiGraph,nlp::GraphModel,option_dict::Dict,partition::Symbol)
    part= get_part(graph,nlp,partition)
    K = length(unique(part))
    option_dict[:schwarz_part] = part
    option_dict[:schwarz_num_parts] = K
end

#prototype optimize using subgraphs
#:linear_solver=>MadNLPSchur,:linear_solver=>MadNLPSchwarz
function optimize!(graph::OptiGraph; partition=:auto, option_dict=Dict{Symbol,Any}(), kwargs...)
    @assert partition in (:auto,:nodes,:subgraphs)
    #parse partition arguments
    if partition == :auto
        partition = has_subgraphs(graph) ? :subgraphs : :nodes
    end
    # K := number of partitions
    # if partition == :subgraphs
    #     K = num_subgraphs(graph)
    # else
    #     K = num_all_nodes(graph)
    # end

    nlp = GraphModel(graph)

    if _get_option(:linear_solver,option_dict,kwargs) == MadNLPSchur
        _set_schur_options!(graph,nlp,option_dict,partition)
    end
    if _get_option(:linear_solver,option_dict,kwargs) == MadNLPSchwarz
        _set_schwarz_options!(graph,nlp,option_dict,partition)
    end

    option_dict[:jacobian_constant] = nlp.ext[:jac_constant]
    option_dict[:hessian_constant] = nlp.ext[:hess_constant]

    #schur and schwarz options get passed through to linear solver initialization
    ips = InteriorPointSolver(nlp; option_dict=option_dict, kwargs...)
    result = optimize!(ips)

    #set optigraph optimizer to InteriorPointSolver
    graph.optimizer = ips

    #update solution
    @blas_safe_threads for k=1:num_all_nodes(graph) #K
        moi_optimizer(nlp.modelnodes[k]).result = MadNLPExecutionStats(
            ips.status,view(result.solution,nlp.ninds[k]),
            ips.obj_val,
            view(result.constraints,nlp.minds[k]),
            ips.inf_du, ips.inf_pr,
            view(result.multipliers,nlp.minds[k]),
            view(result.multipliers_L,nlp.ninds[k]),
            view(result.multipliers_U,nlp.ninds[k]),
            ips.cnt.k, ips.nlp.counters,ips.cnt.total_time)
            # TODO: quick hack to specify to JuMP that the
            # model is not dirty (so we do not run in `OptimizeNotCalled`
            # exception).
            nlp.modelnodes[k].model.is_model_dirty = false
    end
end

# original optimize!
# #TODO: use subgraphs for partitions
# #WHY?:
# more direct interface, don't have to aggregate which can be too much memory and time-consuming
# easier to convey structure to MadNLP
# function optimize!(graph::OptiGraph; option_dict = Dict{Symbol,Any}(), kwargs...)
#     graph.objective_function.constant = 0.
#     nlp = GraphModel(graph)
#     K = num_all_nodes(graph)
#     # setup schwarz options if option given
#     if (haskey(kwargs,:schwarz_custom_partition) && kwargs[:schwarz_custom_partition]) ||
#         (haskey(option_dict,:schwarz_custom_partition) && option_dict[:schwarz_custom_partition])
#         part= get_part(graph,nlp)
#         option_dict[:schwarz_part] = part
#         option_dict[:schwarz_num_parts] = num_all_nodes(graph)
#
#     elseif (haskey(kwargs,:schur_custom_partition) && kwargs[:schur_custom_partition]) ||
#         (haskey(option_dict,:schur_custom_partition) && option_dict[:schur_custom_partition])
#
#         part= get_part(graph,nlp)
#         part[part.>K].=0
#         option_dict[:schur_part] = part
#         option_dict[:schur_num_parts] = K
#     end
#
#     option_dict[:jacobian_constant] = nlp.ext[:jac_constant]
#     option_dict[:hessian_constant] = nlp.ext[:hess_constant]
#
#     #pass linear solver here
#     ips = InteriorPointSolver(nlp;option_dict=option_dict,kwargs...)
#     result = optimize!(ips)
#
#     #set optigraph optimizer to InteriorPointSolver
#     graph.optimizer = ips
#
#     #graph.objective_function.constant = graph.optimizer.obj_val
#
#     @blas_safe_threads for k=1:K
#         moi_optimizer(nlp.modelnodes[k]).result = MadNLPExecutionStats(
#             ips.status,view(result.solution,nlp.ninds[k]),
#             ips.obj_val,
#             view(result.constraints,nlp.minds[k]),
#             ips.inf_du, ips.inf_pr,
#             view(result.multipliers,nlp.minds[k]),
#             view(result.multipliers_L,nlp.ninds[k]),
#             view(result.multipliers_U,nlp.ninds[k]),
#             ips.cnt.k, ips.nlp.counters,ips.cnt.total_time)
#     end
# end

#Retrieve objective value from MadNLP.InteriorPointSolver
MathOptInterface.get(ips::InteriorPointSolver,::MathOptInterface.ObjectiveValue) = ips.obj_val
