import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling, ASAPooling, GCNConv, HANConv,\
    global_max_pool as gmp, global_add_pool, global_sort_pool as mp2, GINConv,global_mean_pool as gap
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os
from torch import nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import math
class CoAttentionLayer(nn.Module):
    def __init__(self, dim, k):
        super(CoAttentionLayer, self).__init__()
        self.k = k
        self.dim = dim
        self.W_m = nn.Linear(dim, k)
        self.W_v = nn.Linear(dim, k)
        self.W_q = nn.Linear(dim, k)
        self.W_h_v = nn.Linear(k, dim)
        self.W_h_q = nn.Linear(k, dim)
        self.softmax=nn.Softmax(dim=-1)
        self.relu=nn.ReLU()
        self.l1=nn.Linear(dim, dim)
        self.l2=nn.Linear(dim, dim)
        self.bn1=nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.lf=nn.Linear(2*dim,dim)


    def forward(self, v0, q0, v1, q1):
        M_0 =  v0*q0

        H_v1 = self.W_v(v1)
        H_q1 = self.W_q(q1)
        H_m = self.W_m(M_0)
        H_v = torch.tanh(H_v1* H_m)
        H_q = torch.tanh(H_q1* H_m)
        alpha_v = F.softmax(self.W_h_v(H_v), dim=-1)
        alpha_q = F.softmax(self.W_h_q(H_q), dim=-1)
        vector1 = alpha_v * v1
        vector2 = alpha_q * q1
        V_0 = vector1 * vector2
        out = torch.cat([vector1,vector2,M_0, V_0], dim=1)
        return out
class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x
class MultiPoolingModule2(torch.nn.Module):
    def __init__(self, num_features):
        super(MultiPoolingModule2, self).__init__()
        self.drop = nn.Dropout(0.1)
        self.pooling_ratio = 0.3
        self.topk_pool1 = SAGPooling(num_features, ratio=self.pooling_ratio, GNN=GCNConv)
        self.b1 = nn.BatchNorm1d(num_features)
        self.b2 =nn.BatchNorm1d(num_features)
        self.b3 = nn.BatchNorm1d(num_features)
        self.b4 = nn.BatchNorm1d(num_features)
        self.b5= nn.BatchNorm1d(num_features)
        self.bf1= nn.LayerNorm(num_features)
        self.bf11 = nn.LayerNorm(num_features)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)
        self.relu=nn.ReLU()
        self.l1=nn.Linear(num_features,num_features)
        self.l11=nn.Linear(num_features,64)
        self.l12=nn.Linear(64,num_features)

    def forward(self, x, edge_index, batch):

        x_topk, edge_index_topk, edge_attr, batch_topk, perm, score = self.topk_pool1(x, edge_index, None, batch)
        x1= x[perm]
        batch = batch[perm]

        x1 = self.drop(torch.stack((self.b1(gmp(x1, batch)), self.b2(gap(x1, batch)),
                            self.b3(global_add_pool(x1, batch))), dim=1))
        x1r=x1
        x1=self.drop(self.l12(self.relu(self.l11(self.drop1(self.bf1(x1))))))
        x1=self.drop(self.bf11(x1+x1r))
        return x1
class MultiPoolingModule(torch.nn.Module):
    def __init__(self, num_features):
        super(MultiPoolingModule, self).__init__()
        self.drop = nn.Dropout(0.1)
        self.b1 = nn.BatchNorm1d(num_features)
        self.b2 =nn.BatchNorm1d(num_features)
        self.b3 = nn.BatchNorm1d(num_features)
        self.b4 = nn.BatchNorm1d(num_features)
        self.b5= nn.BatchNorm1d(num_features)
        self.bf1= nn.LayerNorm(num_features)
        self.bf11 = nn.LayerNorm(num_features)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)
        self.relu=nn.ReLU()
        self.l1=nn.Linear(num_features,num_features)
        self.l11=nn.Linear(num_features,64)
        self.l12=nn.Linear(64,num_features)

    def forward(self, x, edge_index, batch):
        x1 = self.drop(torch.stack((self.b1(gmp(x, batch)), self.b2(gap(x, batch)),
                                    self.b3(global_add_pool(x, batch))), dim=1))
        x1r=x1
        x1=self.drop(self.l12(self.relu(self.l11(self.drop1(self.bf1(x1))))))
        x1=self.drop(self.bf11(x1+x1r))
        return x1
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(maxlen, d_model, device=device)
        self.encoding.requires_grad_(False)

        pos = torch.arange(0, maxlen, device=device)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class LayerNorm(nn.Module):
    def __init__(self, d_model, max_len,eps=1e-10):
        super(LayerNorm, self).__init__()
        self.bn=nn.BatchNorm1d(max_len)
        self.Relu=nn.ReLU()

    def forward(self, x):
        x=self.Relu(self.bn(x))
        return x
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        if mask is not None:
            # mask = torch.tril(torch.ones(time, time, dtype=bool))
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)

        output = self.w_combine(score)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob,max_len) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model,max_len)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model,max_len)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)

        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.drop2(x)
        x = self.norm2(x + _x)
        return x
class Encoder(nn.Module):
    def __init__(
        self,
        env_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layer,
        drop_prob,
        device,
    ):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(
            env_voc_size, d_model, max_len, drop_prob, device
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, drop_prob,max_len)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x
class CKDTA(torch.nn.Module):
    def __init__(self, n_output=1, embed_dim=128, num_features_xd=128,  output_dim=128,
                 dropout_rate=0.1):
        super(CKDTA, self).__init__()
        self.dropnode_rate = 0.2
        self.norm_mode = 'PN-SI'
        self.norm_scale = 1
        self.n_output = n_output

        self.gcn_drug1 = GCNConv(num_features_xd, num_features_xd)
        self.bn_gcn_drug1 = PairNorm(self.norm_mode, self.norm_scale)
        self.bn_at_drug1 = PairNorm(self.norm_mode, self.norm_scale)

        self.gcn_drug2 = GCNConv(num_features_xd, num_features_xd)
        self.bn_gcn_drug2 = PairNorm(self.norm_mode, self.norm_scale)
        self.bn_at_drug2= PairNorm(self.norm_mode, self.norm_scale)

        self.gcn_drug3 = GCNConv(num_features_xd, num_features_xd)
        self.bn_gcn_drug3 = PairNorm(self.norm_mode, self.norm_scale)
        self.bn_at_drug3 = PairNorm(self.norm_mode, self.norm_scale)


        self.fc_druggraph = torch.nn.Linear(3
                                            * num_features_xd, output_dim)
        self.bn_fc_druggraph = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmod = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.lcd = nn.Linear(768, 768*3)
        self.lct = nn.Linear(embed_dim, embed_dim)
        self.lnd = nn.LayerNorm(768*3)
        self.lnt = nn.LayerNorm(embed_dim)
        in_channels = {'atom': 128, 'attribute': 128,'frag': 128}
        out_channels = 128
        metadata = (['atom', 'attribute','frag'], [('atom', 'bond', 'atom'), ('attribute', 'related_to', 'atom'),('atom', 'part_of', 'frag'), ('frag', 'bond', 'frag'),('attribute', 'related_to', 'atom','part_of', 'frag'),('attribute', 'related_to', 'atom','part_of', 'frag','bond', 'frag'),('attribute', 'related_to', 'atom','bond','atom')])
        self.conv1 = HANConv(in_channels, out_channels, metadata=metadata, heads=4,dropout=0.1)
        self.conv2 =HANConv(in_channels, out_channels, metadata=metadata, heads=4,dropout=0.1)
        self.conv3 = HANConv(in_channels, out_channels, metadata=metadata, heads=4,dropout=0.1)

        self.fc_prot1 = nn.Linear(((1) ) * embed_dim, embed_dim*3)
        self.bn_fc_prot1 = nn.BatchNorm1d( embed_dim*3)
        self.fc_prot2 = nn.Linear( 3*embed_dim, output_dim)
        self.bn_fc_prot2 = nn.BatchNorm1d(output_dim)
        self.fc_smile1 = nn.Linear(768*3,  embed_dim)
        self.bn_fc_smile1 = nn.BatchNorm1d( embed_dim)
        self.fc_smile2 = nn.Linear( embed_dim, output_dim)
        self.bn_fc_smile2 = nn.BatchNorm1d(output_dim)
        self.num_features_xt = 54
        self.gcn_target1 = GCNConv(self.num_features_xt, self.num_features_xt)
        self.bn_gcn_target1 = PairNorm(self.norm_mode, self.norm_scale)
        self.gcn_target2 = GCNConv(self.num_features_xt, self.num_features_xt)
        self.bn_gcn_target2 = PairNorm(self.norm_mode, self.norm_scale)
        self.gcn_target3 = GCNConv(self.num_features_xt, self.num_features_xt)
        self.bn_gcn_target3 = PairNorm(self.norm_mode, self.norm_scale)
        self.fc_targetgraph = torch.nn.Linear(3* 54, output_dim)
        self.bn_fc_targetgraph = nn.BatchNorm1d(output_dim)
        self.fc_targetgraph1 = torch.nn.Linear(output_dim, output_dim)
        self.fc_concat1= nn.Linear(8*embed_dim, 1024)
        self.bn_fc_concat1 = nn.BatchNorm1d(1024)
        self.fc_concat2 = nn.Linear(1024, 512)
        self.bn_fc_concat2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, self.n_output)
        self.b1 = nn.BatchNorm1d(num_features_xd)
        self.b2 = nn.BatchNorm1d(num_features_xd)
        self.b3 = nn.BatchNorm1d(num_features_xd)
        self.b4 = nn.BatchNorm1d(num_features_xd)
        self.b5 = nn.BatchNorm1d(num_features_xd)
        self.b6 = nn.BatchNorm1d(num_features_xd)
        self.b7 = nn.BatchNorm1d(num_features_xd)
        self.b8 = nn.BatchNorm1d(num_features_xd)
        self.mu1 = MultiPoolingModule2(54)
        self.mu2 = MultiPoolingModule2(54)
        self.mu3 = MultiPoolingModule2(54)
        self.mu21 = MultiPoolingModule(num_features_xd)
        self.mu22 = MultiPoolingModule(num_features_xd)
        self.mu23 = MultiPoolingModule(num_features_xd)
        self.co = CoAttentionLayer(dim=128,k=196)
        self.bn_gin_mt1 = PairNorm(self.norm_mode, self.norm_scale)
        self.bn_gin_mt2 = PairNorm(self.norm_mode, self.norm_scale)
        self.bn_gin_mt3 = PairNorm(self.norm_mode, self.norm_scale)
        self.l1=nn.Linear(128,128)
        self.l2 = nn.Linear(128, 128)
        self.ln2 = torch.nn.LayerNorm(num_features_xd)
        self.ln1 = torch.nn.LayerNorm(num_features_xd)
        self.ct = Encoder(
            env_voc_size=29,
            max_len=1001,
            d_model=128,
            ffn_hidden=128*3,
            n_head=2,
            n_layer=2,
            drop_prob=0,
            device=device
        )
        config = AutoConfig.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        config.num_hidden_layers = 3
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0
        config.classifier_dropout = 0.0
        self.xd_token = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.xd_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", config=config)

    def _apply_dropnode(self, node_features, batch):
        unique_batches = batch.unique()
        mask = torch.zeros(node_features.size(0), device=node_features.device)

        for batch_id in unique_batches:
            batch_mask = batch == batch_id
            num_nodes_in_graph = batch_mask.sum().item()

            # Randomly decide which nodes to keep
            keep_mask = torch.rand(num_nodes_in_graph, device=node_features.device) > self.dropnode_rate
            mask[batch_mask] = keep_mask.float()

        # Apply the mask to the node features
        mask = mask.view(-1, 1)  # Reshape for broadcasting
        dropped_features = node_features * mask

        return dropped_features

    def forward(self, DrugData, TargetData):
        x_dict, edge_index_dict,attr = DrugData.x_dict, DrugData.edge_index_dict,DrugData.edge_attr_dict
        batch = DrugData['frag'].batch
        batch_atom = DrugData['atom'].batch
        batch_attri=DrugData['attribute'].batch
        edge_index = DrugData[('frag', 'bond', 'frag')].edge_index
        xdseq=DrugData.smilesequ
        (node_type_1, atom_xd), (node_type_2, attribute_xd), (node_type_3, graph_xd) = list(x_dict.items())[:3]
        attribute_xdo=attribute_xd
        x_dict = self.conv1(x_dict, edge_index_dict)
        (node_type_1, atom_xd), (node_type_2, _), (node_type_3, graph_xd) = list(x_dict.items())[:3]
        atom_xd=self._apply_dropnode(atom_xd, batch_atom)
        graph_xd=self.relu(self.bn_gcn_drug1(graph_xd))
        graph_xd=self._apply_dropnode(graph_xd, batch)
        graph_xd1 = graph_xd
        x_dict = {
            node_type_1: atom_xd,
            node_type_2: attribute_xd,
            node_type_3: graph_xd
        }
        x_dict = self.conv2(x_dict, edge_index_dict)
        (node_type_1, atom_xd), (node_type_2, _), (node_type_3, graph_xd) = list(x_dict.items())[:3]
        graph_xd = self.relu(self.bn_gcn_drug2(graph_xd))
        graph_xd = self._apply_dropnode(graph_xd, batch)
        graph_xd2 = graph_xd
        atom_xd = self._apply_dropnode(atom_xd, batch_atom)
        x_dict = {
            node_type_1: atom_xd,
            node_type_2: attribute_xd,
            node_type_3: graph_xd
        }
        x_dict = self.conv3(x_dict, edge_index_dict)
        (node_type_1, atom_xd), (node_type_2, _), (node_type_3, graph_xd) = list(x_dict.items())[:3]
        graph_xd = self._apply_dropnode(graph_xd, batch)
        graph_xd3 = graph_xd
        tx, target_edge_index, tar_batch = TargetData.x, TargetData.edge_index, TargetData.batch
        target = TargetData.target
        graph_xd1=self.mu21(graph_xd1,edge_index, batch)
        graph_xd2 = self.mu22(graph_xd2, edge_index, batch)
        graph_xd3=self.mu23(graph_xd3, edge_index, batch)
        graph_xt = self.relu(self.bn_gcn_target1(self.gcn_target1(tx, target_edge_index)))
        graph_xt1 = self.mu1(graph_xt, target_edge_index, tar_batch)
        graph_xt = self._apply_dropnode(graph_xt, tar_batch)
        graph_xt = self.relu(self.bn_gcn_target2(self.gcn_target2(graph_xt, target_edge_index)))
        graph_xt2 = self.mu2(graph_xt, target_edge_index, tar_batch)
        graph_xt = self._apply_dropnode(graph_xt, tar_batch)
        graph_xt = self.relu(self.bn_gcn_target3(self.gcn_target3(graph_xt, target_edge_index)))
        graph_xt3 = self.mu3(graph_xt, target_edge_index, tar_batch)
        cls_token_id = 28
        batch_size, seq_len =target.shape
        cls_token = torch.full((batch_size, 1), cls_token_id, device=target.device,
                               dtype=target.dtype)
        target= torch.cat([cls_token, target], dim=1)
        conv_xt = self.ct(target,None)
        conv_xt= conv_xt[:, 0, :]
        xd_emb = self.xd_token.batch_encode_plus(
            xdseq,
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt"
        ).to(device)
        conv_xd = self.xd_model(**xd_emb)
        conv_xd = conv_xd.pooler_output
        conv_xd = self.dropout(self.lnd(self.relu(self.lcd(conv_xd))))
        conv_xd = self.dropout(self.bn_fc_smile1(self.relu(self.fc_smile1(conv_xd))))
        conv_xd = self.dropout(self.bn_fc_smile2(self.relu(self.fc_smile2(conv_xd))))
        conv_xt = self.dropout(self.lnt(self.relu(self.lct(conv_xt))))
        conv_xt = self.dropout(self.bn_fc_prot1(self.relu(self.fc_prot1(conv_xt))))
        conv_xt = self.dropout(self.bn_fc_prot2(self.relu(self.fc_prot2(conv_xt))))
        graph_xt=(graph_xt1+graph_xt2+graph_xt3)/3
        graph_xd=(graph_xd1+graph_xd2+graph_xd3)/3
        graph_xt = self.dropout(self.relu(self.bn_fc_targetgraph(self.fc_targetgraph(graph_xt.reshape(-1, 3 * 54)))))
        graph_xd = self.dropout(self.relu(self.bn_fc_druggraph(self.fc_druggraph(graph_xd.reshape(-1, 3 * 128)))))
        out = self.co(graph_xt, graph_xd, conv_xt, conv_xd)
        xc = torch.cat(
            [graph_xt, graph_xd, conv_xt, conv_xd, out], dim=1)
        xc = self.dropout(self.relu(self.bn_fc_concat1(self.fc_concat1(xc))))
        xc = self.dropout(self.bn_fc_concat2(self.relu(self.fc_concat2(xc))))
        out = self.out(xc)
        return out