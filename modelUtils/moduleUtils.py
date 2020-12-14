import torch
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F
import copy
import math
#############
MASK_VALUE = -1e9
#############

######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
######++++++++++++++++++++++++++++Different score functions+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class BiLinear(nn.Module):
    def __init__(self, project_dim: int, args):
        super(BiLinear, self).__init__()
        self.inp_drop = nn.Dropout(args.input_drop)
        self.bilinear_map = nn.Bilinear(in1_features=project_dim, in2_features=project_dim, out_features=1, bias=False)

    def forward(self, query_emb: T, doc_emb: T):
        q_embed = self.inp_drop(query_emb)
        doc_embed = self.inp_drop(doc_emb)
        scores = self.bilinear_map(doc_embed, q_embed).squeeze(dim=-1)
        return scores

class DotProduct(nn.Module):
    def __init__(self, args):
        super(DotProduct, self).__init__()
        self.inp_drop = nn.Dropout(args.input_drop)

    def forward(self, source_emb: T, target_emb: T):
        s_embed = self.inp_drop(source_emb)
        t_embed = self.inp_drop(target_emb)
        scores = torch.matmul(s_embed, t_embed.transpose(-1,-2))
        return scores

class MLP(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_input, d_mid, d_out, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(d_input, d_mid)
        self.w_2 = nn.Linear(d_mid, d_out)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def init(self):
        nn.init.kaiming_uniform_(self.w_1.weight.data)
        nn.init.kaiming_uniform_(self.w_2.weight.data)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
######++++++++++++++++++++++++++++++++++++++++Greedy tree structure decoder+++++++++++++++++++++++++++++++++++++++++++++
######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GraphScoreFunction(nn.Module):
    def __init__(self, d_model):
        super(GraphScoreFunction, self).__init__()
        self.d_model = d_model
        self.linears = clones(nn.Linear(self.d_model, self.d_model), 2)
        self.init()

    def init(self):
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight.data)

    def forward(self, query: T, key: T, mask:T=None) -> T:
        if mask is not None:
            mask = mask.unsqueeze(dim=1)
        batch_size = query.shape[0]
        query, key = [l(x).view(batch_size, -1, self.d_model)
                                                      for l, x in zip(self.linears, (query, key))]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(self.d_model)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        scores = torch.tanh(scores)
        return scores

def graph_decoder(score_matrix: T, start_idx: int, score_mask:T=None, mask: T=None):
    ################################################################################
    graph_nodes = [start_idx]
    graph_edges = []
    ################################################################################
    sub_graph_scores = []
    scores = score_matrix.clone()
    if score_mask is not None:
        assert len(score_mask.shape) == 1
        scores = scores.masked_fill(score_mask == 0, MASK_VALUE)
        mask = mask.unsqueeze(dim=-1)
        scores = scores.masked_fill(score_mask == 0, MASK_VALUE)
    ################################################################################
    score_dim = scores.shape[-1]
    scores.fill_diagonal_(fill_value=MASK_VALUE)
    scores[:, start_idx] = MASK_VALUE
    ################################################################################
    if mask is not None:
        assert len(mask.shape) == 1
        graph_num = mask.sum().detach().item()
        scores = scores.masked_fill(mask == 0, MASK_VALUE)
        mask = mask.unsqueeze(dim=-1)
        scores = scores.masked_fill(mask == 0, MASK_VALUE)
    else:
        ################################################################################
        candidate_scores = scores[graph_nodes]
        max_idx = torch.argmax(candidate_scores)
        max_idx = max_idx.detach().item()
        row_idx, col_idx = max_idx // score_dim, max_idx % score_dim
        orig_row_idx = graph_nodes[row_idx]
        sub_graph_scores.append(score_matrix[orig_row_idx, col_idx])
        graph_edges.append((orig_row_idx, col_idx))
        ################################################################################
        graph_num = score_dim
    ################################################################################

    while True:
        candidate_scores = scores[graph_nodes]
        max_idx = torch.argmax(candidate_scores)
        max_idx = max_idx.detach().item()
        row_idx, col_idx = max_idx // score_dim, max_idx % score_dim
        if candidate_scores[row_idx, col_idx] == MASK_VALUE:
            break
        orig_row_idx = graph_nodes[row_idx]
        ################################################################################
        if mask is not None:
            graph_edges.append((orig_row_idx, col_idx))
            graph_nodes.append(col_idx)
            sub_graph_scores.append(score_matrix[orig_row_idx, col_idx])
            if len(graph_nodes) == graph_num:
                break
        else:
            if score_matrix[orig_row_idx, col_idx] < 0 or len(graph_nodes) >= graph_num:
                break
            else:
                graph_edges.append((orig_row_idx, col_idx))
                graph_nodes.append(col_idx)
                sub_graph_scores.append(score_matrix[orig_row_idx, col_idx])
        ################################################################################
        scores[:, col_idx] = MASK_VALUE
        ################################################################################
    if len(sub_graph_scores)==0:
        print(score_matrix, mask)
    sub_graph_score = torch.stack(sub_graph_scores).sum()
    return graph_nodes, graph_edges, sub_graph_score

def graph_decoding_score(score_matrix: T, start_idx: int, mask: T):
    neg_subgraph = graph_decoder(score_matrix=score_matrix, start_idx=start_idx)
    pos_subgraph = graph_decoder(score_matrix=score_matrix, start_idx=start_idx, mask=mask)
    pn_pair_score = torch.stack([pos_subgraph[-1], neg_subgraph[-1]])
    return pn_pair_score
######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GraphPPRScoreFunc(nn.Module):
    def __init__(self, d_model, alpha=0.15, hop_num=6):
        super(GraphPPRScoreFunc, self).__init__()
        self.d_model = d_model
        self.linears = clones(nn.Linear(self.d_model, self.d_model), 2)
        self.alpha = alpha
        self.hop_num = hop_num
        self.init()

    def init(self):
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight.data)

    def forward(self, query: T, key: T, mask: T=None) -> T:
        if mask is not None:
            mask = mask.unsqueeze(dim=1)
        batch_size = query.shape[0]
        query, key = [l(x).view(batch_size, -1, self.d_model)
                                                      for l, x in zip(self.linears, (query, key))]
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(self.d_model)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        print(scores)
        p_attn = F.softmax(scores, dim=-1)
        return p_attn

    def ppr_propogation(self, p_attn):

        return
######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # print(query.shape, key.shape, value.shape)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    # print('attn{}'.format(p_attn))
    if dropout is not None:
        p_attn = dropout(p_attn)
    res=torch.matmul(p_attn, value)
    # print('res shape {} pa {}'.format(res.shape, p_attn.shape))
    return res, p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model, attn_drop=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // heads
        self.h = heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=attn_drop)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # print(self.attn.shape)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def init(self):
        nn.init.kaiming_uniform_(self.w_1.weight.data)
        nn.init.kaiming_uniform_(self.w_2.weight.data)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class TransformerLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model: int, heads: int, attn_drop: float = 0.1, input_drop: float = 0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model=d_model, heads=heads, attn_drop=attn_drop)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=4*d_model, dropout=input_drop)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(input_drop)

    def forward(self, x, x_mask: T = None):
        x_res = self.self_attn.forward(query=x, key=x, value=x, mask=x_mask)
        x_res = x_res + self.dropout(self.self_attn_norm(x_res))
        x_res = x_res + self.dropout(self.ff_norm(self.feed_forward(x_res)))
        return x_res

######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
######++++++++++++++++++++++++++++++++++++++++Transformer+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TransformerModule(nn.Module):
    def __init__(self, layer_num: int, d_model: int, heads: int, attn_drop: float = 0.1, input_drop: float = 0.1):
        super(TransformerModule, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            transformer_i = TransformerLayer(d_model=d_model, heads=heads, attn_drop=attn_drop, input_drop=input_drop)
            self.layers.append(transformer_i)

    def forward(self, x: T, mask: T=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.rand((2, 3, 8))

    x_mask = torch.LongTensor([[1,1,0], [1,1,0]]).view(2,3).unsqueeze(-1)

    print('x {}'.format(x.shape))

    transformer = TransformerModule(layer_num=1, d_model=8, heads=1, attn_drop=0, input_drop=0)

    y = transformer.forward(x, mask=x_mask)

    print('y {}'.format(y.shape))

    # ppr_score = GraphPPRScoreFunc(d_model=128)
    # attn = ppr_score.forward(x, x, mask=mask)
    # print(attn)
    # print(attn)
    # print(x)
    # x_mask = torch.randint(0,1, size=(2, 3)).bool()
    # print(x_mask.shape)
    # transformer = TransformerModule(d_model=128, heads=1, layer_num=2)
    # x = transformer.forward(x, x_mask)
    # print(x)


    # print(x)
    # transformer = Transformer(d_model=128, heads=4, attn_drop=0.0, input_drop=0.0)
    # x_mask = torch.ones((2,5,5), dtype=torch.bool)
    # x_mask[:,:,4] = False
    # x_mask[:,4,:] = False
    # y = transformer.forward(x, x, x, x_mask)
    # print(y)
    # print()