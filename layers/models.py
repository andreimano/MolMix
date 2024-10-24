import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import TfModule
from layers.encoders import get_edge_encoder, get_node_encoder
from layers.gemnet import GemNetT
from layers.gnn import GINEConv
from layers.schnet import MySchNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISABLE = False
MODE = "default"

dataset_output_dim = {
    "zinc": 1,
    "qm9": 1,
    "esol": 1,
    "freesolv": 1,
    "lipo": 1,
    "bace": 1,
    "kraken": 4,
    "kraken_B5": 1,
    "kraken_L": 1,
    "kraken_burB5": 1,
    "kraken_burL": 1,
    "drugs": 4,
    "drugs_energy": 1,
    "drugs_ip": 1,
    "drugs_ea": 1,
    "drugs_chi": 1,
}


def get_model(args, device):
    model = ProbGT(
        n_gnn_layers=args.model.n_gnn_layers,
        n_tf_heads=args.model.n_tf_heads,
        n_tf_layers=args.model.n_tf_layers,
        tf_dropout=args.model.tf_dropout,
        hidden_dim=args.model.hidden_dim,
        tf_hidden_dim=args.model.tf_hidden_dim,
        agg_tf_heads=args.model.agg_tf_heads if args.model.agg_tf_heads else 0,
        use_1d=args.use_smiles if hasattr(args, "use_smiles") else True,
        use_2d=args.use_2d if hasattr(args, "use_2d") else True,
        use_3d=args.multimodal if hasattr(args, "multimodal") else True,
        args=args,
    ).to(device)

    return model


class LayerPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        import math

        super(LayerPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x, smiles_indices):
        x = x + self.pe[smiles_indices]
        return self.dropout(x)


class AsciiCharEncoder(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(AsciiCharEncoder, self).__init__()

        self.vocab = string.printable
        self.vocab_size = len(self.vocab) + 2
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.pad_token = self.vocab_size - 1
        self.cls_token = self.vocab_size - 2

        self.n_heads = n_heads

        self.embed = nn.Embedding(self.vocab_size, hidden_dim)

    def forward(self, x: list):
        smiles_tokenized = []
        for string in x:
            tokenized = [self.char_to_idx[char] for char in string]
            smiles_tokenized.append(torch.tensor(tokenized))

        batch_tokenized = torch.cat(smiles_tokenized).to(DEVICE)

        return self.embed(batch_tokenized)


class ModelAscii(nn.Module):
    def __init__(self, args):
        super(ModelAscii, self).__init__()

        self.vocab = string.printable
        self.vocab_size = len(self.vocab) + 2
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.pad_token = self.vocab_size - 1
        self.cls_token = self.vocab_size - 2

        self.ascii_tokenizer = AsciiCharEncoder(
            args.model.hidden_dim, args.model.agg_tf_heads
        )
        self.pos_embed = LayerPositionalEncoding(
            args.model.hidden_dim, dropout=0.0, max_len=1024
        )

        self.model_ascii = TfModule(
            dim_h=args.model.hidden_dim,
            num_heads=args.model.agg_tf_heads,
            num_layers=args.model.agg_tf_layers,
            dropout=args.model.tf_dropout,
        )

    def forward(self, data, cumsum_seq, smiles_indices, max_len, train=True):
        smiles_string = data.smiles
        smiles_tensor = self.ascii_tokenizer(smiles_string)
        smiles_tensor = self.pos_embed(smiles_tensor, smiles_indices)

        x_ascii = self.model_ascii(smiles_tensor, cumsum_seq, max_len)  # [:, 0, :]

        return x_ascii


class Model3D(nn.Module):
    def __init__(self, args):
        super(Model3D, self).__init__()

        # should write a get_model function
        if args.model.model_3d == "schnet":
            self.model_3d = MySchNet(
                hidden_channels=args.model.hidden_dim, num_filters=args.model.hidden_dim
            )

        elif args.model.model_3d == "gemnet":
            gemnet_cfg = {
                "num_spherical": 7,
                "num_radial": 6,
                "num_blocks": 4,
                "emb_size_atom": args.model.hidden_dim,
                "emb_size_edge": args.model.hidden_dim,
                "emb_size_trip": 64,
                "emb_size_rbf": 16,
                "emb_size_cbf": 16,
                "emb_size_bil_trip": 64,
                "num_before_skip": 1,
                "num_after_skip": 1,
                "num_concat": 1,
                "num_atoms": 1,
                "num_atom": 2,
                "bond_feat_dim": 0,
            }
            # Max atomic number 300 might be too much, default on schnet too
            self.model_3d = GemNetT(max_atomic_num=100, **gemnet_cfg)

        else:
            raise NotImplementedError

    def forward(self, data, cumsum_seq, train=True):
        model_3d_tf_outputs = []
        for i in range(len(data.pos)):
            x_3d = self.model_3d(data.z, data.pos[i], data.z_batch)
            model_3d_tf_outputs.append(x_3d)

        chunk_lengths = cumsum_seq[1:] - cumsum_seq[:-1]
        centroids_3d_stack = torch.stack(model_3d_tf_outputs)
        centroids_3d_split = centroids_3d_stack.split(chunk_lengths.tolist(), dim=1)
        centroids_3d_split = [
            centroid.reshape(centroid.shape[0] * centroid.shape[1], centroid.shape[2])
            for centroid in centroids_3d_split
        ]
        centroids_3d = torch.cat(centroids_3d_split, dim=0)

        return centroids_3d


class Model2D(nn.Module):

    def __init__(self, args):
        super(Model2D, self).__init__()

        self.node_encoder = get_node_encoder(
            args.dataset,
            args.model.hidden_dim,
            lap_dim=args.model.lap_dim if hasattr(args.model, "lap_dim") else 0,
            rwse_dim=args.model.rwse_dim if hasattr(args.model, "rwse_dim") else 0,
        )
        self.edge_encoder = get_edge_encoder(args.dataset, args.model.hidden_dim)

        self.gnn_convs = nn.ModuleList(
            [
                GINEConv(
                    self.edge_encoder,
                    in_channels=args.model.hidden_dim,
                    out_channels=args.model.hidden_dim,
                )
                for _ in range(args.model.n_gnn_layers)
            ]
        )

    def forward(self, data, cumsum_seq, train=True):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.node_encoder(data)

        gnn_tf_outputs = []

        for i in range(len(self.gnn_convs)):
            initial_input = x
            x = self.gnn_convs[i](x, edge_index, edge_attr)
            x += initial_input
            x = F.gelu(x)  # This one is important

            gnn_tf_outputs.append(x.clone())

        chunk_lengths = cumsum_seq[1:] - cumsum_seq[:-1]
        centroids_2d_stack = torch.stack(gnn_tf_outputs)
        centroids_2d_split = centroids_2d_stack.split(chunk_lengths.tolist(), dim=1)
        centroids_2d_split = [
            centroid.reshape(centroid.shape[0] * centroid.shape[1], centroid.shape[2])
            for centroid in centroids_2d_split
        ]
        centroids_2d = torch.cat(centroids_2d_split, dim=0)

        return centroids_2d


def cat_list_varlen(tensors_list, cumsums_list, cumsum_final):
    full_len = cumsum_final[-1]
    first_ptr = tensors_list[0]
    cat_tensor = torch.empty(
        (full_len, first_ptr.shape[1]), device=first_ptr.device, dtype=first_ptr.dtype
    )
    full_chunks = cumsum_final[1:] - cumsum_final[:-1]
    before_chunks = torch.zeros_like(cumsum_final[:-1])
    for tensor, cumsum_seq in zip(tensors_list, cumsums_list):
        cumsum_seq = cumsum_seq.to(tensor.device, non_blocking=True)
        copy_indexes = torch.arange(
            tensor.shape[0], dtype=torch.int64, device=tensor.device
        )
        chunk_lengths = cumsum_seq[1:] - cumsum_seq[:-1]
        # first, shift for all the chunks from previous tensors
        # for example, if i have the sequences [0, 1, 2, 5, 6, 10, 11], [3, 4, 7, 8, 9, 12, 13], i have cumsum [0, 7, 14]
        # further, the first sequence is [0, 1, 2], [3, 4] and cumsum = [0, 3, 5]
        # and the second is [5, 6], [7, 8, 9] with cumsum = [0, 2, 5]
        # when writing [5, 6], i have to take in account the previous [0, 1, 2]
        # similarily for [7, 8, 9], but for further sequences i also have to be careful about their absolute position in the full tensor
        copy_indexes += torch.repeat_interleave(before_chunks, chunk_lengths).to(
            device=tensor.device
        )
        before_chunks += chunk_lengths
        # then shift to the fixed positions where each sequence starts
        # i subtract the chunk lengths to compensate for the fact that current chunk lengths are already included in the range
        # for example, if i have the sequences [0, 1, 2, 5, 6, 10, 11], [3, 4, 7, 8, 9, 12, 13], i have cumsum [0, 7, 14]
        # further, the first sequence is [0, 1, 2], [3, 4] and cumsum = [0, 3, 5]
        # to correctly write [3, 4] at positions [7, 8],  i have the copy_indexes = [0, 1, 2, 3, 4]
        # to which i add 0 on the first 3 and 7 - 3 on the last two to get [0, 1, 2, 7, 8]
        # which are the correct positions
        repeated_chunks = torch.repeat_interleave(
            cumsum_seq[1:-1], chunk_lengths[1:]
        ).to(tensor.device)
        chunk_shift = torch.repeat_interleave(cumsum_final[1:-1], chunk_lengths[1:]).to(
            tensor.device
        )
        copy_indexes[cumsum_seq[1] :] += chunk_shift - repeated_chunks
        cat_tensor = cat_tensor.scatter(
            0, copy_indexes.unsqueeze(1).expand_as(tensor), tensor
        )

    return cat_tensor


class ProbGT(torch.nn.Module):
    def __init__(
        self,
        n_gnn_layers=5,
        n_tf_heads=5,
        n_tf_layers=5,
        tf_dropout=0.1,
        hidden_dim=32,
        tf_hidden_dim=64,
        agg_tf_heads=3,
        use_1d=True,
        use_2d=True,
        use_3d=True,
        args=None,
    ):
        super(ProbGT, self).__init__()

        self.tf_hidden_dim = tf_hidden_dim
        self.n_tf_heads = n_tf_heads
        self.agg_tf_heads = agg_tf_heads

        self.use_1d = use_1d
        self.use_3d = use_3d
        self.use_2d = use_2d

        if self.use_1d:
            self.model_1d = ModelAscii(args)

        if self.use_3d:
            self.model_3d = Model3D(args)

        if self.use_2d:
            self.model_2d = Model2D(args)

        self.separate_readout = (
            args.separate_readout if hasattr(args, "separate_readout") else False
        )

        # Downstream multimodal transformer
        self.tf = TfModule(
            dim_h=self.tf_hidden_dim,
            num_heads=n_tf_heads,
            num_layers=n_tf_layers,
            dropout=tf_dropout,
            new_arch=args.model.new_arch if hasattr(args.model, "new_arch") else True,
            expansion=(
                args.model.expansion if hasattr(args.model, "expansion") else 8 / 3
            ),
        )

        # Special tokens
        self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_dim)).to(DEVICE)
        self.cls_token = nn.Parameter(torch.empty(1, hidden_dim)).to(DEVICE)
        self.sep_token = nn.Parameter(torch.empty(1, hidden_dim)).to(DEVICE)
        # Modality positional embeddings
        self.pos_emb_2d = nn.Parameter(torch.empty(1, hidden_dim)).to(DEVICE)
        self.pos_emb_3d = nn.Parameter(torch.empty(1, hidden_dim)).to(DEVICE)
        self.pos_emb_1d = nn.Parameter(torch.empty(1, hidden_dim)).to(DEVICE)

        # havent found a cleaner way to do this, maybe in the dataloader object
        self.cumsum_1d = torch.zeros(args.batch_size + 1, dtype=torch.int32)
        self.cumsum_2d = torch.zeros(args.batch_size + 1, dtype=torch.int32)
        self.cumsum_3d = torch.zeros(args.batch_size + 1, dtype=torch.int32)
        self.full_cumsum = torch.zeros(args.batch_size + 1, dtype=torch.int32)
        self.ones_cumsum = torch.ones(args.batch_size, dtype=torch.int32).cumsum_(0)
        self.max_seqlen_1d = 0
        self.max_seqlen_3d = 0
        self.max_seqlen_2d = 0
        self.max_seqlen_full = 0
        self.n_gnn_layers = n_gnn_layers
        self.bs = args.batch_size

        if hasattr(args, "separate_readout") and args.separate_readout:
            self.readout = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(tf_hidden_dim, tf_hidden_dim // 2),
                        nn.GELU(),
                        nn.Linear(tf_hidden_dim // 2, 1),
                    )
                    for _ in range(dataset_output_dim[args.dataset])
                ]
            )
        else:
            self.readout = nn.Sequential(
                nn.Linear(tf_hidden_dim, tf_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(tf_hidden_dim // 2, dataset_output_dim[args.dataset]),
            )
        self.init_tokens()

    def init_tokens(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.1)
        nn.init.normal_(self.sep_token, mean=0.0, std=0.1)
        nn.init.normal_(self.pos_emb_3d, mean=0.0, std=0.1)
        nn.init.normal_(self.pos_emb_2d, mean=0.0, std=0.1)
        nn.init.normal_(self.pos_emb_1d, mean=0.0, std=0.1)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.1)

    def build_cumsum_smiles(self, data, seqnum):
        self.cumsum_1d *= 0
        for seq in range(seqnum):
            self.cumsum_1d[seq + 1] = len(data.smiles[seq])
        self.cumsum_1d.cumsum_(0)
        self.max_seqlen_1d = torch.max(
            self.cumsum_1d[1:] - self.cumsum_1d[:-1]
        ).item()  # since no one is on cuda i think this won't hurt that much for smiles

    def build_cumsum_3d(self, data, seqnum):
        self.cumsum_3d *= 0
        for seq in range(seqnum):
            self.cumsum_3d[seq + 1] = (data.z_batch == seq).sum()
        self.cumsum_3d.cumsum_(0)
        self.max_seqlen_3d = torch.max(self.cumsum_3d[1:] - self.cumsum_3d[:-1]).item()

    def build_cumsum_2d(self, data, seqnum):
        self.cumsum_2d *= 0
        for seq in range(seqnum):
            self.cumsum_2d[seq + 1] = (data.batch == seq).sum()
        self.cumsum_2d.cumsum_(0)
        self.max_seqlen_2d = torch.max(self.cumsum_2d[1:] - self.cumsum_2d[:-1]).item()

    def forward(self, data, train=True):
        seqnum = data.z_batch[-1] + 1
        if self.use_1d:
            self.build_cumsum_smiles(data, seqnum)
            chunk_lengths = self.cumsum_1d[1:] - self.cumsum_1d[:-1]
            indices_1d = torch.cat([torch.arange(l) for l in chunk_lengths]).to(
                DEVICE, non_blocking=True
            )

            tokens_1d_no_sep = self.model_1d(
                data,
                self.cumsum_1d[: seqnum + 1].to(DEVICE),
                indices_1d,
                self.max_seqlen_1d,
                train,
            )

            # okay so the copy logic is pretty involved
            # first i allocate memory for the flattened batch + sep tokens
            tokens_1d = torch.empty(
                (tokens_1d_no_sep.shape[0] + seqnum, tokens_1d_no_sep.shape[1]),
                device=DEVICE,
                dtype=tokens_1d_no_sep.dtype,
            )
            # then i write the sep tokens, they can be written at the end of every chunk + all the previous inserted sep tokens
            # so for example, for chunk 1 there is no previous sep token, so i just write it at the end of chunk 1
            # chunk 2 has the sep token from chunk 1 and its own sep token, so i write it at the end of chunk 2 + 1
            # and so on, using ones_cumsum
            tokens_1d[
                self.cumsum_1d[1 : seqnum + 1]
                + torch.cat([torch.tensor([0]), self.ones_cumsum[: seqnum - 1]])
            ] = self.sep_token
            # then i create a tensor with entries from 0, 1, ..., all_batch_tokens
            copy_indexes = torch.arange(tokens_1d_no_sep.shape[0], dtype=torch.int64)
            # repeat trick to correctly add the shift amount to each chunk in copy_indexes
            repeated_sums = torch.repeat_interleave(
                self.ones_cumsum[: seqnum - 1], chunk_lengths[1:seqnum]
            )
            # 1st chunk does not have to be shifted
            copy_indexes[self.cumsum_1d[1] :] += repeated_sums
            # finally, do an index copy
            tokens_1d = tokens_1d.scatter(
                0,
                copy_indexes.to(DEVICE).unsqueeze(1).expand_as(tokens_1d_no_sep),
                tokens_1d_no_sep,
            )
            self.cumsum_1d[1:] += self.ones_cumsum  # shift everybody for the sep tokens

        if self.use_3d:
            self.build_cumsum_3d(data, seqnum)
            tokens_3d_no_sep = self.model_3d(data, self.cumsum_3d[: seqnum + 1], train)

            self.cumsum_3d[1:] *= len(data.pos)

            chunk_lengths = self.cumsum_3d[1:] - self.cumsum_3d[:-1]

            tokens_3d = torch.empty(
                (tokens_3d_no_sep.shape[0] + seqnum, tokens_3d_no_sep.shape[1]),
                device=DEVICE,
                dtype=tokens_3d_no_sep.dtype,
            )
            tokens_3d[
                self.cumsum_3d[1 : seqnum + 1]
                + torch.cat([torch.tensor([0]), self.ones_cumsum[: seqnum - 1]])
            ] = self.sep_token

            copy_indexes = torch.arange(tokens_3d_no_sep.shape[0], dtype=torch.int64)
            repeated_sums = torch.repeat_interleave(
                self.ones_cumsum[: seqnum - 1], chunk_lengths[1:seqnum]
            )
            copy_indexes[self.cumsum_3d[1] :] += repeated_sums
            tokens_3d = tokens_3d.scatter(
                0,
                copy_indexes.to(DEVICE).unsqueeze(1).expand_as(tokens_3d_no_sep),
                tokens_3d_no_sep,
            )
            self.cumsum_3d[1:] += self.ones_cumsum  # shift everybody for the sep tokens

        if self.use_2d:
            self.build_cumsum_2d(data, seqnum)
            centroids_2d_no_sep = self.model_2d(
                data, self.cumsum_2d[: seqnum + 1], train
            )

            self.cumsum_2d[1:] *= self.n_gnn_layers

            chunk_lengths = self.cumsum_2d[1:] - self.cumsum_2d[:-1]

            tokens_2d = torch.empty(
                (centroids_2d_no_sep.shape[0] + seqnum, centroids_2d_no_sep.shape[1]),
                device=DEVICE,
                dtype=centroids_2d_no_sep.dtype,
            )
            tokens_2d[
                self.cumsum_2d[1 : seqnum + 1]
                + torch.cat([torch.tensor([0]), self.ones_cumsum[: seqnum - 1]])
            ] = self.sep_token
            copy_indexes = torch.arange(centroids_2d_no_sep.shape[0], dtype=torch.int64)
            repeated_sums = torch.repeat_interleave(
                self.ones_cumsum[: seqnum - 1], chunk_lengths[1:seqnum]
            )
            try:
                copy_indexes[self.cumsum_2d[1] :] += repeated_sums
            except:
                # import pdb
                # pdb.set_trace()
                pass
            tokens_2d = tokens_2d.scatter(
                0,
                copy_indexes.to(DEVICE).unsqueeze(1).expand_as(centroids_2d_no_sep),
                centroids_2d_no_sep,
            )
            self.cumsum_2d[1:] += self.ones_cumsum  # shift everybody for the sep tokens

        centroids_list = []
        cumsums_list = []

        if self.use_3d:
            tokens_3d += self.pos_emb_3d
            centroids_list.append(tokens_3d)
            cumsums_list.append(self.cumsum_3d[: seqnum + 1])

        if self.use_1d:
            tokens_1d += self.pos_emb_1d
            centroids_list.append(tokens_1d)
            cumsums_list.append(self.cumsum_1d[: seqnum + 1])

        if self.use_2d:
            tokens_2d += self.pos_emb_2d
            centroids_list.append(tokens_2d)
            cumsums_list.append(self.cumsum_2d[: seqnum + 1])

        self.full_cumsum = (
            torch.stack(cumsums_list)
            .sum(0)
            .to(torch.int32, non_blocking=True)
            .to(DEVICE, non_blocking=True)
        )
        self.max_seqlen_full = torch.max(
            self.full_cumsum[1:] - self.full_cumsum[:-1]
        ).item()
        tokens_no_cat = cat_list_varlen(
            centroids_list, cumsums_list, self.full_cumsum[: seqnum + 1]
        )

        # append [CLS]
        chunk_lengths = self.full_cumsum[1:] - self.full_cumsum[:-1]
        tokens = torch.empty(
            (tokens_no_cat.shape[0] + seqnum, tokens_no_cat.shape[1]),
            device=DEVICE,
            dtype=tokens_no_cat.dtype,
        )
        tokens[
            self.full_cumsum[:seqnum]
            + torch.cat([torch.tensor([0]), self.ones_cumsum[: seqnum - 1]]).to(
                device=DEVICE, non_blocking=True
            )
        ] = self.cls_token
        copy_indexes = torch.arange(
            tokens_no_cat.shape[0], dtype=torch.int64, device=DEVICE
        )
        repeated_sums = torch.repeat_interleave(
            self.ones_cumsum[:seqnum].to(DEVICE, non_blocking=True), chunk_lengths
        )
        copy_indexes += repeated_sums
        tokens = tokens.scatter(
            0,
            copy_indexes.to(DEVICE).unsqueeze(1).expand_as(tokens_no_cat),
            tokens_no_cat,
        )
        self.full_cumsum[1:] += self.ones_cumsum[:seqnum].to(
            DEVICE, non_blocking=True
        )  # shift everybody for the sep tokens

        out = self.tf(tokens, self.full_cumsum[: seqnum + 1], self.max_seqlen_full)
        out = out[self.full_cumsum[:seqnum]]  # get the [cls] tokens

        if self.separate_readout:
            out = [readout(out) for readout in self.readout]
            out = torch.cat(out, dim=1)
        else:
            out = self.readout(out)

        if len(out.shape) == 2 and out.shape[1] == 1 and len(data.y.shape) == 1:
            out = out.squeeze(1)

        return out
