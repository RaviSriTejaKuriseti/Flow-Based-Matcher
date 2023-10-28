
import numpy as np
import torch
from ortools.graph.python import min_cost_flow
from functools import reduce
from operator import add
import os,time



def match_with_dummy_gt(C,max_matches,dummy_weight):
    
    smcf = min_cost_flow.SimpleMinCostFlow()
    num_queries=C.shape[0]
    num_gt=C.shape[1]

    if(num_gt==0):
        U,V,c=torch.tensor([]).to(torch.int64),torch.tensor([]).to(torch.int64),torch.tensor([]).to(torch.int64)
        return (U,V)

    
    num_gt_new=num_gt+1
    sink_ind=num_queries+num_gt_new+1
    gt_start=num_queries+1

        
    R=reduce(add,[[1+i]*(num_gt) for i in range(num_queries)])
    S=R+[0]*num_queries+[i+1 for i in range (num_queries)]+[gt_start+i for i in range(num_gt_new)]
    E=[gt_start+i for i in range (num_gt)]*(num_queries)+[i+1 for i in range (num_queries)]+[sink_ind-1]*num_queries+[sink_ind]*num_gt_new
    Costs=list((C[:,:num_gt].numpy().ravel())*10**4)+[0]*num_queries+[dummy_weight]*num_queries+[0]*num_gt_new

    start_nodes=np.array(S)
    end_nodes=np.array(E)
    capacities=np.array([1]*(num_queries)+[1]*(num_queries*num_gt_new)+[max_matches-1]*num_gt+[(num_gt)*(max_matches-1)]*1)
    unit_costs=np.array(Costs)

    # Define an array of supplies at each node.
    supplies=[max_matches*num_gt]+[0]*(num_queries)+[-1]*num_gt+[0]+[(1-max_matches)*num_gt]


    # Add arcs, capacities and costs in bulk using numpy.
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, capacities, unit_costs)

    # Add supply for each nodes.
    smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

    # Find the min cost flow.
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        print('There was an issue with the min cost flow input.')
        print(f'Status: {status}')
        exit(1)
    solution_flows = smcf.flows(all_arcs)


    Q_ind=[]
    T_ind=[]
    C_ind=[]

    indices=np.where(solution_flows!=0)
    arcs=all_arcs[indices]
    req_costs = unit_costs[indices]
    for arc,ct in zip(arcs,req_costs):
        h=smcf.head(arc)
        t=smcf.tail(arc)
        if(t!=0 and h!=sink_ind):
            Q_ind.append(t-1)
            T_ind.append(h-1-num_queries)
            C_ind.append(ct)


    Q_ind=torch.tensor(Q_ind).to(torch.int64)
    T_ind=torch.tensor(T_ind).to(torch.int64)
    return (Q_ind,T_ind)


def many_to_one_without_dummy(C,max_matches):
    smcf = min_cost_flow.SimpleMinCostFlow()
    num_queries=C.shape[0]
    num_gt=C.shape[1]

    if(num_gt==0):
        U,V=torch.tensor([]).to(torch.int64),torch.tensor([]).to(torch.int64)
        return (U,V)

    
    sink_ind=num_queries+num_gt+1
    gt_start=num_queries+1

    
    
    R=reduce(add,[[1+i]*(num_gt) for i in range(num_queries)])
    S=R+[0]*num_queries+[gt_start+i for i in range(num_gt)]
    E=[gt_start+i for i in range (num_gt)]*(num_queries)+[i+1 for i in range (num_queries)]+[sink_ind]*num_gt
    Costs=list((C[:,:num_gt].numpy().ravel())*10**4)+[0]*num_queries+[0]*num_gt

    start_nodes=np.array(S)
    end_nodes=np.array(E)
    capacities=np.array([1]*(num_queries)+[1]*(num_queries*num_gt)+[max_matches-1]*num_gt)
    # capacities=np.array([1]*(num_queries)+[1]*(num_queries*num_gt)+[max_matches]*num_gt)
    unit_costs=np.array(Costs)

    
    # Define an array of supplies at each node.
    supplies=[max_matches*num_gt]+[0]*(num_queries)+[-1]*num_gt+[(1-max_matches)*num_gt]

    # Add arcs, capacities and costs in bulk using numpy.
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, capacities, unit_costs)

    # Add supply for each nodes.
    smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)

    # Find the min cost flow.
    status = smcf.solve()
    if status != smcf.OPTIMAL:
        print('There was an issue with the min cost flow input.')
        print(f'Status: {status}')
        exit(1)
    solution_flows = smcf.flows(all_arcs)


    Q_ind=[]
    T_ind=[]

    indices=np.where(solution_flows!=0)
    arcs=all_arcs[indices]
    for arc in arcs:
        h=smcf.head(arc)
        t=smcf.tail(arc)
        if(t!=0 and h!=sink_ind):
            Q_ind.append(t-1)
            T_ind.append(h-1-num_queries)


    Q_ind=torch.tensor(Q_ind).to(torch.int64)
    T_ind=torch.tensor(T_ind).to(torch.int64)
    return (Q_ind,T_ind)

