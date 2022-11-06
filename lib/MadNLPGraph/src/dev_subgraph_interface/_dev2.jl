using MadNLP
using MadNLPGraph
import MadNLP: get_lcon, get_ucon
using Plasmo

function simple_graph()
    n_nodes=100
    M = 200
    d = sin.((0:M*n_nodes).*pi/100)

    graph = OptiGraph()
    @optinode(graph,nodes[1:n_nodes])

    #Node models
    nodecnt = 1
    for (i,node) in enumerate(nodes)
        @variable(node, x[1:M])
        @variable(node, -1<=u[1:M]<=1)
        @constraint(node, dynamics[i in 1:M-1], x[i+1] == x[i] + u[i])
        @objective(node, Min, sum(x[i]^2 - 2*x[i]*d[i+(nodecnt-1)*M] for i in 1:M) +
            sum(u[i]^2 for i in 1:M))
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
kwargs = Dict(:linear_solver=>MadNLPSchur,  :schur_custom_partition=>false, :schur_num_parts=>8)#, :print_level=>MadNLP.ERROR)
MadNLP.optimize!(graph; kwargs...)

optimizer = graph.ext[:ips]
two_stage_partition = optimizer.linear_solver.tsp
part = two_stage_partition.part
nlp = optimizer.nlp
n = nlp.ext[:n]
m = nlp.ext[:m]
p = nlp.ext[:p]
ninds = nlp.ninds #node variables
minds = nlp.minds #node constraints
pinds = nlp.pinds #link constraints

part[1:n]         #variable membership
part[n+1:n+m]     #constraint membership
part[n+m+1:end]   #link constraint membership

findall(part .== 0) #there are zeros for the 7 cuts, and the 7 corresponding attached nodes

#you can also get the incidence matrix Sungho partitions:
csc = optimizer.linear_solver.csc
MadNLPGraph.Graph(csc) #this is a lightgraph. It adds an edge for every variable constraint connection (i.e. for every nonzero in the kkt system)
