import dgl
import torch

def graph_batcher(graphs):
    batch_graph = dgl.batch(graphs=graphs)
    return batch_graph

def graph_unbatcher(batch_graph):
    graphs = dgl.unbatch(g=batch_graph)
    return graphs


if __name__ == '__main__':
    g1 = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))) ### graph, from --> to
    g1.edata['e_id'] = torch.LongTensor([0,1,2])
    g1.ndata['h'] = torch.arange(0,4,1)
    g2 = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])))
    g2.edata['e_id'] = torch.LongTensor([0, 1, 2, 3])
    g2.ndata['h'] = torch.arange(0,3,1)
    batch_graph = graph_batcher([g1, g2])
    print(batch_graph.ndata['h'])
    # print(batch_graph)

    # graph_unbatcher(batch_graph)
    # #
    # # print(g1)
    # # print(g2)