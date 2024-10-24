import torch
from ogb.graphproppred.mol_encoder import AtomEncoder as OGB_AtomEncoder
from ogb.graphproppred.mol_encoder import BondEncoder as OGB_BondEncoder
from torch_geometric.nn import MLP


def get_node_encoder(dataset, hidden_dim, lap_dim=0, rwse_dim=0):
    if rwse_dim > 0 or lap_dim > 0:
        return FeatureEncoder(hidden_dim, hidden_dim, dataset, lap_dim, rwse_dim)
    if dataset.lower() == "zinc":
        return ZINCAtomEncoder(hidden_dim)
    if dataset.lower() in ["esol", "bace", "freesolv", "lipo"]:
        return MyOGBAtomEncoder(hidden_dim)
    if dataset.lower().startswith("kraken"):
        return MyOGBAtomEncoder(hidden_dim)
    if dataset.lower().startswith("drugs"):
        return MyOGBAtomEncoder(hidden_dim)
    else:
        raise NotImplementedError


def get_edge_encoder(dataset, hidden_dim):
    if dataset.lower() == "zinc":
        return ZINCBondEncoder(hidden_dim)
    if dataset.lower() in ["esol", "bace", "freesolv", "lipo"]:
        return torch.nn.LazyLinear(hidden_dim)
    if dataset.lower().startswith("kraken"):
        return torch.nn.LazyLinear(hidden_dim)
    if dataset.lower().startswith("drugs"):
        return torch.nn.LazyLinear(hidden_dim)
    else:
        raise NotImplementedError


class ZINCBondEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCBondEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, edge_attr):
        if edge_attr is not None:
            return self.embedding(edge_attr)
        else:
            return None


class ZINCAtomEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(ZINCAtomEncoder, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=21, embedding_dim=hidden)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, data):
        return self.embedding(data.x)


class MyOGBAtomEncoder(torch.nn.Module):
    def __init__(self, hidden):
        super(MyOGBAtomEncoder, self).__init__()
        self.embedding = OGB_AtomEncoder(hidden)

    def forward(self, data):
        return self.embedding(data.x)


class LapPENodeEncoder(torch.nn.Module):
    # https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/laplace_pos_encoder.py
    """Laplace Positional Embedding node encoder.
    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.
    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_in, dim_emb, pecfg, expand_x=True):
        super().__init__()

        dim_pe = pecfg.dim_pe  # Size of Laplace PE embedding
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        max_freqs = pecfg.max_freqs  # Num. eigenvectors (frequencies)

        if dim_emb - dim_pe < 0:  # formerly 1, but you could have zero feature size
            raise ValueError(
                f"LapPE size {dim_pe} is too large for "
                f"desired embedding size of {dim_emb}."
            )

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = torch.nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if pecfg.raw_norm_type is None or pecfg.raw_norm_type == "None":
            raw_norm = torch.nn.Identity()
        elif pecfg.raw_norm_type.lower() == "batchnorm":
            raw_norm = torch.nn.BatchNorm1d(max_freqs)
        else:
            raise ValueError

        self.pe_encoder = torch.nn.Sequential(
            raw_norm,
            MLP([max_freqs] + (n_layers - 1) * [2 * dim_pe] + [dim_pe], act="gelu"),
        )

    def forward(self, x, batch):
        if not hasattr(batch, "EigVecs"):
            raise ValueError(
                "Precomputed eigen values and vectors are "
                f"required for {self.__class__.__name__}; "
                "set config 'posenc_LapPE.enable' to True"
            )
        pos_enc = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(pos_enc.size(1), device=pos_enc.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            pos_enc = pos_enc * sign_flip.unsqueeze(0)

        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors)

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors)
        # pos_enc = (self.pe_encoder(eigvecs) + self.pe_encoder(-eigvecs)) * 0.5  # (Num nodes) x dim_pe
        pos_enc = self.pe_encoder(pos_enc)

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(x)
        else:
            h = x
        # Concatenate final PEs to input embedding
        x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        return x


class RWSENodeEncoder(torch.nn.Module):
    # https://github.com/rampasek/GraphGPS/blob/main/graphgps/encoder/kernel_pos_encoder.py
    """Configurable kernel-based Positional Encoding node encoder.
    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.
    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.
    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = "RWSE"  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_in, dim_emb, pecfg, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(
                f"{self.__class__.__name__} has to be "
                f"preconfigured by setting 'kernel_type' class"
                f"variable before calling the constructor."
            )

        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        num_rw_steps = pecfg.kernel
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        n_layers = pecfg.layers

        if dim_emb - dim_pe < 0:  # formerly 1, but you could have zero feature size
            raise ValueError(
                f"PE dim size {dim_pe} is too large for "
                f"desired embedding size of {dim_emb}."
            )

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = torch.nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        if norm_type == "batchnorm":
            self.raw_norm = torch.nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        self.pe_encoder = MLP(
            [num_rw_steps] + (n_layers - 1) * [2 * dim_pe] + [dim_pe], act="gelu"
        )

    def forward(self, x, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(
                f"Precomputed '{pestat_var}' variable is "
                f"required for {self.__class__.__name__}; set "
                f"config 'posenc_{self.kernel_type}.enable' to "
                f"True, and also set 'posenc.kernel.times' values"
            )

        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(x)
        else:
            h = x
        # Concatenate final PEs to input embedding
        x = torch.cat((h, pos_enc), 1)
        return x


class FeatureEncoder(torch.nn.Module):

    def __init__(self, hidden, dataset: str, lap_dim: int, rwse_dim: int):
        super(FeatureEncoder, self).__init__()

        lin_hidden = hidden
        if lap_dim > 0:
            lin_hidden -= lap_dim
        if rwse_dim > 0:
            lin_hidden -= rwse_dim

        assert lin_hidden > 0

        self.linear_embed = get_node_encoder(dataset, lin_hidden)

        # rwse:
        #     kernel: 20
        #     layers: 2
        #     dim_pe: 32
        #     raw_norm_type: 'BatchNorm'

        # lap:
        #     max_freqs: 4
        #     dim_pe: 32
        #     layers: 1
        #     raw_norm_type: null

        lap_encoder_cfg = {
            "dim_pe": lap_dim,
            "layers": 1,
            "max_freqs": 4,
            "raw_norm_type": None,
        }

        rw_encoder_cfg = {
            "kernel": 20,
            "layers": 2,
            "dim_pe": 32,
            "raw_norm_type": "BatchNorm",
        }

        if lap_dim > 0:
            self.lap_encoder = LapPENodeEncoder(
                hidden,
                hidden - (rwse_dim if rwse_dim > 0 else 0),
                lap_encoder_cfg,
                expand_x=False,
            )
        else:
            self.lap_encoder = None

        if rwse_dim > 0:
            self.rw_encoder = RWSENodeEncoder(
                hidden, hidden, rw_encoder_cfg, expand_x=False
            )
        else:
            self.rw_encoder = None

    def forward(self, data):
        x = self.linear_embed(data)
        if self.lap_encoder is not None:
            x = self.lap_encoder(x, data)
        if self.rw_encoder is not None:
            x = self.rw_encoder(x, data)
        return x
