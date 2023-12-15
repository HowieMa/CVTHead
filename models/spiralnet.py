import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import openmesh as om
import numpy as np 
import pickle 

from sklearn.neighbors import KDTree


"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Spiral Conv Enoder >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
"""


def to_sparse(spmat):
    # (M, N)  sparse matrix of numpy
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))



class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        # (B, V, 3)
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1).to(x.device))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1).to(x.device))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


def Pool(x, trans, dim=1):
    # (B, V0, C)
    # trans: (V1, V0), tensor
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)       # (V1, 1), all one
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out


class SpiralDecoder(nn.Module):
    def __init__(self, emb_dim, spiral_indices, up_transform):
        super(SpiralDecoder, self).__init__()

        self.spiral_indices = spiral_indices    # 5k, 1k, 
        self.up_transform = up_transform        # (5023, 1256), (1256, 314)

        num_layer = len(up_transform)

        # decoder
        self.de_layers = nn.ModuleList()
        for idx in range(num_layer):
            self.de_layers.append(
                SpiralDeblock(emb_dim, emb_dim, self.spiral_indices[num_layer - idx - 1]))
        
        self.de_layers.append(
            SpiralConv(emb_dim, emb_dim, self.spiral_indices[0]))   # 5023 - 5023
    
    def forward(self, x):
        # x: (B, V_coarse=314, C)
        num_layers = len(self.de_layers)    # 3
        num_features = num_layers - 1       # 2

        for i, layer in enumerate(self.de_layers):
            if i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i - 1].to(x.device))
            else:
                x = layer(x)

        return x

"""
>>>>>>>>>>>>>>>>>>>>>>>>> Mesh Transform Model >>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""


def _next_ring(mesh, last_ring, other):
    res = []

    def is_new_vertex(idx):
        return (idx not in last_ring and idx not in other and idx not in res)

    for vh1 in last_ring:
        vh1 = om.VertexHandle(vh1)
        after_last_ring = False
        for vh2 in mesh.vv(vh1):
            if after_last_ring:
                if is_new_vertex(vh2.idx()):
                    res.append(vh2.idx())
            if vh2.idx() in last_ring:
                after_last_ring = True
        for vh2 in mesh.vv(vh1):
            if vh2.idx() in last_ring:
                break
            if is_new_vertex(vh2.idx()):
                res.append(vh2.idx())
    return res


def extract_spirals(mesh, seq_length, dilation=1):
    # output: spirals.size() = [N, seq_length]
    spirals = []
    for vh0 in mesh.vertices():
        reference_one_ring = []
        for vh1 in mesh.vv(vh0):
            reference_one_ring.append(vh1.idx())
        spiral = [vh0.idx()]
        one_ring = list(reference_one_ring)
        last_ring = one_ring
        next_ring = _next_ring(mesh, last_ring, spiral)
        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(mesh, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            kdt = KDTree(mesh.points(), metric='euclidean')
            spiral = kdt.query(np.expand_dims(mesh.points()[spiral[0]],
                                              axis=0),
                               k=seq_length * dilation,
                               return_distance=False).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(spiral[:seq_length * dilation][::dilation])
    return spirals


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    # face: (F, 3); seq_length: 9; vertices (V, 3)
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1     # V
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals


def get_flame_transform(transform_fp):
    dilation = [1, 1, 1, 1]
    seq_length = [9, 9, 9, 9]
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')
    """
    - vertices(5): (5023, 3), (1256, 3), (314, 3), (79, 3), (20, 3)
    - face(5): (9976, 3), (2464, 3), (597, 3), (139, 3), (34, 3)
    - adj(5): (5023, 5023), (1256, 1256), (314, 314), (79, 79), (20, 20)
    - down_transform (4): (1256, 5023), (314, 1256), (79, 314), (20, 79)
    - uptransform(4): (5023, 1256), (1256, 314), (314, 79), (79, 20)
    """
    spiral_indices_list = [
        preprocess_spiral(tmp['face'][idx], seq_length[idx],tmp['vertices'][idx], dilation[idx])
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [to_sparse(down_transform) for down_transform in tmp['down_transform'] ]
    up_transform_list = [to_sparse(up_transform) for up_transform in tmp['up_transform']]
    return spiral_indices_list, down_transform_list, up_transform_list


def get_coarse_mesh_decoder(emb_dim=32, transform_fp = "../assets/transform.pkl", down_degree=2):
    # template_vertices, (1, V, 3)

    # >>>>>>>>>>>>>>>>> get down-sample level >>>>>>>>>>>>>>>>>>>
    dilation = [1, 1, 1, 1]
    seq_length = [9, 9, 9, 9]
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        preprocess_spiral(tmp['face'][idx], seq_length[idx],tmp['vertices'][idx], dilation[idx])
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [to_sparse(down_transform) for down_transform in tmp['down_transform'] ]
    up_transform_list = [to_sparse(up_transform) for up_transform in tmp['up_transform']]
    # (5023, 3), (1256, 3), (314, 3), (79, 3), (20, 3)

    spiral_indices_list = spiral_indices_list[: down_degree]
    down_transform_list = down_transform_list[ : down_degree]
    up_transform_list = up_transform_list[: down_degree]
    
    # >>>>>>>>>>>>>>>> get coase-to-fine decoder >>>>>>>>>>>>>>>>>> 
    mesh_decoder = SpiralDecoder(emb_dim=emb_dim, spiral_indices=spiral_indices_list, up_transform=up_transform_list)
    return mesh_decoder, down_transform_list


def downsample_vertices(vertices, down_transform_list):
    coarse_vertices = vertices.clone()
    for transform in down_transform_list:
        coarse_vertices = Pool(coarse_vertices, transform.to(vertices.device))      # (B, V_coarse, 3)
    return coarse_vertices


# Unit testing
if __name__ == "__main__":
    
    transform_fp = "../assets/transform.pkl"

    v_template = torch.randn(1, 5023, 3)

    # v_template = torch.arange(1, 5023+1).unsqueeze(0).unsqueeze(-1)
    # # print(v_template)
    mesh_decoder, down_transform_list = get_coarse_mesh_decoder(32, transform_fp)

    v_feat = torch.randn(1, 314, 32)

    v_fine_feat = mesh_decoder(v_feat)
    # print(down_transform_list[-1].shape[0])

    v_coarse = downsample_vertices(v_template, down_transform_list)
    print(v_coarse.shape)
    # spiral_indices_list, down_transform_list, up_transform_list = get_flame_transform(transform_fp)
    # print("spiral_indices_list ", len(spiral_indices_list))
    # print("down_transform_list", len(down_transform_list))
