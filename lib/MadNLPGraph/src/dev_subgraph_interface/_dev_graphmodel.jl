# development file for new GraphModel that supports subgraphs
# TODO: finish this file!

import Plasmo: OptiGraph, OptiNode, OptiEdge, all_nodes, all_edges, all_variables, num_all_nodes, getlinkconstraints, getnode, num_variables, num_constraints
import MathOptInterface

import JuMP: _create_nlp_block_data, set_optimizer, GenericAffExpr
import MathOptInterface: get
# import MadNLP
import Ipopt

# pushfirst!(LOAD_PATH,"/home/jordan/.julia/dev/Plasmo")

#function GraphModel(graph::OptiGraph)
function simple_graph()
    n_nodes=100
    M = 10
    d = sin.((0:M*n_nodes).*pi/10)

    graph = OptiGraph()
    @optinode(graph,nodes[1:n_nodes])

    #Node models
    nodecnt = 1
    for (i,node) in enumerate(nodes)
        @variable(node, x[1:M])
        @variable(node, -1<=u[1:M]<=1)
        @constraint(node, dynamics[i in 1:M-1], x[i+1] == x[i] + u[i])
        @objective(node, Min, sum(x[i]^2 - 2*x[i]*d[i+(nodecnt-1)*M] for i in 1:M) + sum(u[i]^2 for i in 1:M))
        nodecnt += 1
    end
    n1 = getnode(graph,1)
    @constraint(n1,n1[:x][1] == 0)
    for i=1:n_nodes-1
        @linkconstraint(graph, nodes[i+1][:x][1] == nodes[i][:x][M] + nodes[i][:u][M],attach=nodes[i+1])
    end
    return graph
end

graph = simple_graph()
node_vec = Vector{Vector{OptiNode}}()
count = 0
for i = 1:10
    v = OptiNode[]
    for j = 1:10
        push!(v,getnode(graph,j + count))
    end
    push!(node_vec,v)
    global count += 10
end
p = Partition(graph,node_vec)
apply_partition!(graph,p)

# ############
# graph2 = simple_graph()
# nlp2 = MadNLPGraph.GraphModel(graph2)
# ############

subs = Plasmo.subgraphs(graph)
linkedges = Plasmo.optiedges(graph)

for sg in subs
    num_variables(sg) == 0 && error("Empty node exists! Delete the empty nodes.")
end
# _caching_optimizer(node::OptiNode) = JuMP.backend(node).optimizer
# moi_optimizer(graph::OptiGraph) = graph.optimizer.optimizer.model
#_caching_optimizer(node::OptiNode) = JuMP.backend(node).optimizer
moi_optimizer(graph::OptiGraph) = backend(graph).optimizer.model

for k=1:length(subs)
    # set_optimizer(subs[k],MadNLP.Optimizer)
    set_optimizer(subs[k], Ipopt.Optimizer)
    if Plasmo.has_nlp_data(graph)
    #if models[k].model.nlp_data !== nothing
        MOI.set(subs[k].optimizer, MOI.NLPBlock(),
                _create_nlp_block_data(subs[k].optimizer))
        empty!(subs[k].optimizer.nlp_data.nlconstr_duals)
    end
    #TODO:
    #an optigraph does not have anything in its model_cache
    #Plasmo.jl needs to support a true model backend with an optigraph
    #MOIU.attach_optimizer(subs[k].optimizer)
    # attach optimizer to subgraph
    MOIU.attach_optimizer(JuMP.backend(subs[k]))
    MOI.initialize(moi_optimizer(subs[k]).nlp_data.evaluator,[:Grad,:Hess,:Jac])
end

K = length(subs)
ns= [num_variables(sg) for sg in subs]
n = sum(ns)
ns_cumsum = cumsum(ns)
ms= [num_all_constraints(sg) + num_link_constraints(sg) for sg in subs]
ms_cumsum = cumsum(ms)
m = sum(ms)

nnzs_hess = [MadNLP.get_nnz_hess(moi_optimizer(sg)) for sg in subs]
nnzs_hess_cumsum = cumsum(nnzs_hess)
nnz_hess = sum(nnzs_hess)

nnzs_jac = [MadNLP.get_nnz_jac(moi_optimizer(sg)) for sg in subs]
nnzs_jac_cumsum = cumsum(nnzs_jac)
nnz_jac = sum(nnzs_jac)

get_nnz_link_jac(linkedge::OptiEdge) = sum(length(linkcon.func.terms) for (ind,linkcon) in linkedge.linkconstraints)
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

for k=1:K
    MadNLP.set_x!(moi_optimizer(subs[k]),view(x,ninds[k]),view(xl,ninds[k]),view(xu,ninds[k]))
    MadNLP.set_g!(moi_optimizer(subs[k]),view(l,minds[k]),view(gl,minds[k]),view(gu,minds[k]))
end


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

for q=1:Q
    set_g_link!(linkedges[q],view(l,pinds[q]),view(gl,pinds[q]),view(gu,pinds[q]))
end

x_index_map = Dict()
for k = 1:K
    offset = 0
    sg = subs[k]
    for node in all_nodes(sg)
        for var in all_variables(node)
            node_var_index = var.index.value
            x_index_map[var] = ninds[k][node_var_index + offset]
        end
        offset += num_variables(node)
    end
end

#this is a little more difficult with subgraphs
# x_index_map = Dict()
# for k = 1:K
#     offset = 0
#     sg = subs[k]
#     for node in all_nodes(sg)
#         moi_opt = JuMP.backend(node).optimizer
#         for var in all_variables(node)
#             node_var_index = moi_opt.model_to_optimizer_map[var.index].value
#             x_index_map[var] = ninds[k][node_var_index + offset]
#         end
#         offset += num_variables(node)
#     end
# end


cnt = 0
g_index_map = Dict(con=> m + (global cnt+=1) for linkedge in linkedges for (ind,con) in linkedge.linkconstraints)

jac_constant = true
hess_constant = true
for node in modelnodes
    j,h = is_jac_hess_constant(moi_optimizer(node))
    jac_constant = jac_constant & j
    hess_constant = hess_constant & h
end

ext = Dict{Symbol,Any}(:n=>n,:m=>m,:p=>p,:ninds=>ninds,:minds=>minds,:pinds=>pinds,
                       :linkedges=>linkedges,:jac_constant=>jac_constant,:hess_constant=>hess_constant)

# return GraphModel(
#     ninds,minds,pinds,nnzs_jac_inds,nnzs_hess_inds,nnzs_link_jac_inds,
#     x_index_map,g_index_map,modelnodes,linkedges,
#     NLPModelMeta(
#         n,
#         ncon = m+p,
#         x0 = x,
#         lvar = xl,
#         uvar = xu,
#         y0 = l,
#         lcon = gl,
#         ucon = gu,
#         nnzj = nnz_jac + nnz_link_jac,
#         nnzh = nnz_hess,
#         minimize = true # graph.objective_sense == MOI.MIN_SENSE
#     ),
#     graph,NLPModelsCounters(),
#     ext
# )
