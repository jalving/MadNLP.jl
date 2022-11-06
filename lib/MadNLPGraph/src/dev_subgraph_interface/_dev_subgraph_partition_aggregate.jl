#this applies a partition to an optigraph and then aggregates it. using this to
#debug issues with direct subgraph partitions

using MadNLP
using MadNLPGraph
import MadNLP: get_lcon, get_ucon
using Plasmo
using LightGraphs

include("_dev_setup.jl")

#################################
#Convert to subgraphs and use subgraph partition
#################################
graph3 = simple_graph()
node_vec = Vector{Vector{OptiNode}}()
count = 0
for i = 1:10
    v = OptiNode[]
    for j = 1:10
        push!(v,getnode(graph3,j + count))
    end
    push!(node_vec,v)
    global count += 10
end
p = Partition(graph3,node_vec)
apply_partition!(graph3,p)

# aggreate the graph based on the partition
graph3,ref = aggregate(graph3,0)

# create the MadNLP GraphModel
nlp3 = MadNLPGraph.GraphModel(graph3)

#get a part vector based on nodes
part3= MadNLPGraph.get_part(graph3, nlp3, :nodes)
option_dict = Dict{Symbol,Any}()
option_dict[:schur_part] = part3
K = length(unique(part3))
option_dict[:schur_num_parts] = K
option_dict[:jacobian_constant] = nlp3.ext[:jac_constant]
option_dict[:hessian_constant] = nlp3.ext[:hess_constant]

# create the interior point solver
ips3 = MadNLP.InteriorPointSolver(nlp3; option_dict=option_dict, :linear_solver=>MadNLPSchur,  :schur_custom_partition=>true);
ls3 = ips3.linear_solver;

ls3.V_0 # first stage nodes
csc3 = ls3.csc #KKT matrix
csc3_0 = ls3.csc_0 #KKT matrix for first stage

#result3 = MadNLP.optimize!(ips3)
#####################################################

ind_cons3 = MadNLP.get_index_constraints(nlp3)
MT = SparseMatrixCSC{Float64, Int32}
kkt3 = MadNLP.SparseKKTSystem{Float64, MT}(nlp3, ind_cons3)
csc3 = kkt3.aug_com


#I thought about using mark_boundary! as opposed to setting the zero entries myself. It didn't work though.
#part = get_part(graph3,nlp3)
#g3 = MadNLPGraph.Graph(csc3)
#mark_boundary!(g3,part)
