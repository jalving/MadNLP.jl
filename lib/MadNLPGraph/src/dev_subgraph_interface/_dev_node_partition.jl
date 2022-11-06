using MadNLP
using MadNLPGraph
import MadNLP: get_lcon, get_ucon
using Plasmo

include("_dev_setup.jl")

#################################
# Use current get_part in MadNLP with node partitions
#################################
graph = simple_graph()
nlp = MadNLPGraph.GraphModel(graph)
part= MadNLPGraph.get_part(graph,nlp,:nodes)

#Run the solver with the given partition
option_dict = Dict{Symbol,Any}()
option_dict[:schur_part] = part
K = length(unique(part))
option_dict[:schur_num_parts] = K
option_dict[:jacobian_constant] = nlp.ext[:jac_constant]
option_dict[:hessian_constant] = nlp.ext[:hess_constant]
ips = MadNLP.InteriorPointSolver(nlp; option_dict=option_dict, :linear_solver=>MadNLPSchur, :schur_custom_partition=>true);
ls = ips.linear_solver;

print(ips.linear_solver.V_0)
result = MadNLP.optimize!(ips)
print(ips.linear_solver.V_0)

#KKT system. square matrix with dims n+m x n+m
ls = ips.linear_solver
csc = ls.csc

#MADNLP graph of the KKT system. remember the diagaonal creates self-edges
g = MadNLPGraph.Graph(csc)

#this finds the first stage nodes
function mark_boundary!(g,part)
    for e in edges(g)
        #if edge cuts across parts AND src is not in part 0 AND dst is not in part 0, THEN: src and dst get set to part 0
        #this effectively puts linking constraints into the 0 (root) partition
        (part[src(e)]!=part[dst(e)] && part[src(e)]!= 0 && part[dst(e)]!= 0) && (part[src(e)]=0; part[dst(e)]=0)
    end
end

#BUG: re-initialize doesn't seem to work for multiple solves. It also seems to be changing the part to the default number of parts

#part is ordered as:
#part = [node_variables;node_constraints;link_constraints]

#NOTE: attached nodes for link constraints in this `part` get set to partition 0
# graph = simple_graph()
# kwargs = Dict(:linear_solver=>MadNLPSchur,  :schur_custom_partition=>true,:print_level=>MadNLP.ERROR)
# MadNLP.optimize!(graph;kwargs...)
# println(graph.objective_function)
#
# nlp = graph.optimizer.nlp
# n = nlp.ext[:n]
# m = nlp.ext[:m]
# p = nlp.ext[:p]
# ninds = nlp.ninds
# minds = nlp.minds
# pinds = nlp.pinds
# two_stage_partition = graph.optimizer.linear_solver.tsp
# part = two_stage_partition.part
