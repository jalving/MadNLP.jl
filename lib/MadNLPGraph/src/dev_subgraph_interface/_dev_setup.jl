# development setup file for supporting subgraph partitions

# using MadNLP
# using MadNLPGraph
# import MadNLP: get_lcon, get_ucon
using Plasmo

#the original get_part function
function get_part(graph,nlp)
    n = nlp.ext[:n]
    m = nlp.ext[:m]
    p = nlp.ext[:p]

    ninds = nlp.ninds
    minds = nlp.minds
    pinds = nlp.pinds

    ind_ineq = findall(get_lcon(nlp).!=get_ucon(nlp))
    l = length(ind_ineq)
    part = Vector{Int}(undef,n+m+l+p)

    #1:2000
    for k=1:length(ninds)
        part[ninds[k]].=k
    end

    #2000:2901
    for k=1:length(minds)
        part[minds[k].+n.+l].=k
    end

    cnt = 0
    for linkedge in nlp.ext[:linkedges]
        for (ind,con) in linkedge.linkconstraints
            cnt+=1
            attached_node_idx = graph.node_idx_map[con.attached_node]
            part[n+l+m+cnt] = attached_node_idx != nothing ? attached_node_idx : error("All the link constraints need to be attached to a node")
        end
    end

    #inequality constraints
    cnt = 0
    for q in ind_ineq
        cnt+=1
        part[n+cnt] = part[n+l+q]
    end
    return part
end

# proposed get_part for subgraphs
function get_part_subgraphs(graph,nlp)
    n = nlp.ext[:n]
    m = nlp.ext[:m]
    p = nlp.ext[:p]

    #map indices to subgraph partitions
    ninds = nlp.ninds
    minds = nlp.minds
    pinds = nlp.pinds

    ninds_subgraphs = []
    minds_subgraphs = []
    node_subgraph_map = Dict()

    var_index_map = zeros(num_all_variables(graph))
    for (i,sub_graph) in enumerate(subgraphs(graph))
        #variables
        n_start = Plasmo.optinode_by_index(sub_graph,1)
        n_end = Plasmo.optinode_by_index(sub_graph,num_all_nodes(sub_graph))
        n_index_start = nlp.ninds[graph[n_start]][1]
        n_index_end = nlp.ninds[graph[n_end]][end]
        push!(ninds_subgraphs,UnitRange(n_index_start,n_index_end))

        #constraints
        m_index_start = nlp.minds[graph[n_start]][1]
        m_index_end = nlp.minds[graph[n_end]][end]
        push!(minds_subgraphs,UnitRange(m_index_start,m_index_end))

        for node in all_nodes(sub_graph)
            node_subgraph_map[graph.node_idx_map[node]] = i
        end
    end

    ind_ineq = findall(get_lcon(nlp) .!= get_ucon(nlp))
    l = length(ind_ineq)
    part = Vector{Int}(undef,n+m+l+p)

    #n
    for k=1:length(ninds_subgraphs)
        part[ninds_subgraphs[k]].=k
    end

    #m : m+n+l
    for k=1:length(minds_subgraphs)
        part[minds_subgraphs[k].+n.+l].=k
    end

    #local link constraints
    #attached nodes should be mapped to their subgraph
    cnt = 0
    for subgraph in Plasmo.subgraphs(graph)
        for linkedge in Plasmo.all_edges(subgraph)
            for (ind,con) in linkedge.linkconstraints
                cnt+=1
                attached_node_idx = graph.node_idx_map[con.attached_node]
                attached_node_idx == nothing && error("All the link constraints need to be attached to a node")
                graph_idx = node_subgraph_map[attached_node_idx]
                part[n+l+m+cnt] = graph_idx
            end
        end
    end
    n_local_links = cnt

    #global link constraints
    #constraint node and the attached node should have 0 entries in part
    cnt = 0
    for linkedge in optiedges(graph)
        for (ind,con) in linkedge.linkconstraints
            cnt+=1
            attached_node_idx = graph.node_idx_map[con.attached_node]
            attached_node_idx == nothing && error("All the link constraints need to be attached to a node")

            terms = collect(linear_terms(con.func))
            attached_vars = [term[2] for term in terms if getnode(term[2]) == con.attached_node]
            attached_ninds_local = [var.index.value for var in attached_vars]
            attached_ninds = [(nlp.ninds[attached_node_idx].start + local_ind - 1) for local_ind in attached_ninds_local]

            #this is not correctly finding the boundary
            #graph_idx = node_subgraph_map[attached_node_idx]
            #part[n+l+m+n_local_links+cnt] = graph_idx

            #setting part values to 0 manually here
            #variables on the attached node will also be set to 0
            part[n+l+m+n_local_links+cnt] = 0
            part[attached_ninds] .= 0
        end
    end

    #inequality indices get mapped to equality indices?
    cnt = 0
    for q in ind_ineq
        cnt+=1
        part[n+cnt] = part[n+l+q]
    end

    return part
end

# sets partitions to the master partition based on boundaries
function mark_boundary!(g, part)
    for e in edges(g)
        #if edge cuts across parts AND src is not in part 0 AND dst is not in part 0, THEN: src and dst get set to part 0
        (part[src(e)]!=part[dst(e)] && part[src(e)]!= 0 && part[dst(e)]!= 0) && (part[src(e)]=0; part[dst(e)]=0)
    end
end

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
