#this applies a partition to an optigraph directly. this should work just like the aggregate version

using MadNLP
using MadNLPGraph
import MadNLP: get_lcon, get_ucon
using Plasmo
using LightGraphs

include("_dev_setup.jl")

#################################
#Convert to subgraphs and use subgraph partition
#################################
graph2 = simple_graph()
node_vec = Vector{Vector{OptiNode}}()
count = 0
for i = 1:10
    v = OptiNode[]
    for j = 1:10
        push!(v,getnode(graph2,j + count))
    end
    push!(node_vec,v)
    global count += 10
end
p = Partition(graph2,node_vec)
apply_partition!(graph2,p)

# create the NLP problem
nlp2 = MadNLPGraph.GraphModel(graph2)

# get a part vector based on subgraphs
# NOTE: part vector is different than aggregate version. need to check partition is correct.
part2 = get_part_subgraphs(graph2, nlp2)

# this is not allowed
# nlp2.ninds = nlp3.ninds
# nlp2.minds = nlp3.minds
# nlp2.pinds = nlp3.pinds


option_dict = Dict{Symbol,Any}()
option_dict[:schur_part] = part2

K = length(unique(part2)) - 1 #we do not include part 0
option_dict[:schur_num_parts] = K
option_dict[:jacobian_constant] = nlp2.ext[:jac_constant]
option_dict[:hessian_constant] = nlp2.ext[:hess_constant]

ips2 = MadNLP.InteriorPointSolver(nlp2; option_dict=option_dict, :linear_solver=>MadNLPSchur,  :schur_custom_partition=>true);
ls2 = ips2.linear_solver;

ls2.V_0
csc2 = ls2.csc
csc2_0 = ls2.csc_0 #this structure looks incorrect

ls2.sws[1].V_0_nz #BUG: this is empty for every solver worker. there is no connection between sub problems and the schur complement.

ind_cons2 = MadNLP.get_index_constraints(nlp2)
MT = SparseMatrixCSC{Float64, Int32}
kkt2 = MadNLP.SparseKKTSystem{Float64, MT}(nlp2, ind_cons2)
csc2 = kkt2.aug_com

#ips2.linear_solver = ls3;

#this seems to fail. might need to adjust ip ninds and minds based on subgraphs?
#result2 = MadNLP.optimize!(ips2)
