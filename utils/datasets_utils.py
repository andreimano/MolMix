from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.transforms import (
    AddRandomWalkPE,
    AddRemainingSelfLoops,
    Compose,
    ToUndirected,
)
from torch_geometric.utils import (
    get_laplacian,
    sort_edge_index,
    to_scipy_sparse_matrix,
    to_smiles,
)
from torch_geometric.utils.sparse import index2ptr
# from torchmetrics.functional import mean_squared_error

from utils.misc import Config

x_map: Dict[str, List[Any]] = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLENE",
        "CHI_SQUAREPLANAR",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map: Dict[str, List[Any]] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
}


def to_rdmol(
    data: "torch_geometric.data.Data",
    kekulize: bool = False,
) -> Any:
    """Converts a :class:`torch_geometric.data.Data` instance to a
    :class:`rdkit.Chem.Mol` instance.

    Args:
        data (torch_geometric.data.Data): The molecular graph data.
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem

    mol = Chem.RWMol()

    assert data.x is not None
    assert data.num_nodes is not None
    assert data.edge_index is not None
    assert data.edge_attr is not None
    for i in range(data.num_nodes):
        atom = Chem.Atom(int(data.x[i, 0]))
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[int(data.x[i, 1])])
        atom.SetFormalCharge(x_map["formal_charge"][int(data.x[i, 3])])
        atom.SetNumExplicitHs(x_map["num_hs"][int(data.x[i, 4])])
        atom.SetNumRadicalElectrons(x_map["num_radical_electrons"][int(data.x[i, 5])])
        atom.SetHybridization(Chem.rdchem.HybridizationType.values[int(data.x[i, 6])])
        atom.SetIsAromatic(bool(data.x[i, 7]))
        mol.AddAtom(atom)

    edges = [tuple(i) for i in data.edge_index.t().tolist()]
    visited = set()

    for i in range(len(edges)):
        src, dst = edges[i]
        if tuple(sorted(edges[i])) in visited:
            continue

        bond_type = Chem.BondType.values[int(data.edge_attr[i, 0])]
        mol.AddBond(src, dst, bond_type)

        # Set stereochemistry:
        stereo = Chem.rdchem.BondStereo.values[int(data.edge_attr[i, 1])]
        if stereo != Chem.rdchem.BondStereo.STEREONONE:
            db = mol.GetBondBetweenAtoms(src, dst)
            db.SetStereoAtoms(dst, src)
            db.SetStereo(stereo)

        # Set conjugation:
        is_conjugated = bool(data.edge_attr[i, 2])
        mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(is_conjugated)

        visited.add(tuple(sorted(edges[i])))

    mol = mol.GetMol()

    if kekulize:
        Chem.Kekulize(mol)

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)

    return mol


i_conformers = 0
import logging


class AugmentWithConformers:
    def __init__(self, num_conformers=200):
        super(AugmentWithConformers, self).__init__()
        self.num_conformers = num_conformers

    def __call__(self, graph: Data):
        global i_conformers

        pos = []
        while len(pos) == 0:
            mol = Chem.MolFromSmiles(graph.smiles)
            molH = Chem.AddHs(mol)
            confIds = AllChem.EmbedMultipleConfs(
                molH, numConfs=self.num_conformers, numThreads=0
            )
            z = torch.tensor(
                [atom.GetAtomicNum() for atom in molH.GetAtoms()], dtype=torch.long
            )
            pos = [
                torch.tensor(conf.GetPositions(), dtype=torch.float)
                for conf in molH.GetConformers()
            ]
            if len(pos) == 0:
                logging.warn(f"Failed to generate conformers, trying again")

        if len(pos) != self.num_conformers and len(pos) != 0:
            logging.warn(
                f"Number of conformers generated is not equal to {self.num_conformers} ({len(pos)}); duplicating."
            )
            repeats = (self.num_conformers // len(pos)) + 1
            pos = (pos * repeats)[: self.num_conformers]

        graph.z = z
        graph.pos = pos
        i_conformers += 1
        logging.warn(f"{i_conformers} molecules processed")
        return graph


class PygWithConformers:
    def __init__(self, num_conformers=200):
        super(PygWithConformers, self).__init__()
        self.num_conformers = num_conformers

    def __call__(self, graph: Data):

        # mol = to_rdmol(graph)
        # smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        # graph.smiles = smiles

        smiles = to_smiles(graph)
        mol = Chem.MolFromSmiles(smiles)

        molH = Chem.AddHs(mol)
        confIds = AllChem.EmbedMultipleConfs(molH, numConfs=self.num_conformers)
        z = torch.tensor(
            [atom.GetAtomicNum() for atom in molH.GetAtoms()], dtype=torch.long
        )
        pos = [
            torch.tensor(conf.GetPositions(), dtype=torch.float)
            for conf in molH.GetConformers()
        ]
        graph.z = z
        graph.pos = pos
        return graph


class AugmentWithPartition:
    """
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/cluster.html#ClusterData
    """

    def __init__(self, num_parts, recursive=True):
        super(AugmentWithPartition, self).__init__()
        self.num_parts = num_parts
        self.recursive = recursive

    def __call__(self, graph: Data):
        row, index = sort_edge_index(
            graph.edge_index, num_nodes=graph.num_nodes, sort_by_row=True
        )
        indptr = index2ptr(row, size=graph.num_nodes)

        cluster = torch.ops.torch_sparse.partition(
            indptr.cpu(),
            index.cpu(),
            None,
            self.num_parts,
            self.recursive,
        ).to(graph.edge_index.device)

        graph.partition = cluster
        return graph


class AugmentWithDumbAttr:
    def __call__(self, graph: Data):
        graph.x = torch.ones(graph.num_nodes, 1, dtype=torch.float)
        graph.edge_attr = torch.ones(graph.num_edges, 1, dtype=torch.float)
        return graph


class RenameLabel:
    # dumb class to rename edge_label to y
    def __call__(self, graph: Data):
        graph.y = graph.edge_label.float()  # for BCE loss
        del graph.edge_label
        return graph


class AddLaplacianEigenvectorPE:
    """
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.AddLaplacianEigenvectorPE.html
    """

    # Number of nodes from which to use sparse eigenvector computation:
    SPARSE_THRESHOLD: int = 100

    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        **kwargs: Any,
    ) -> None:
        self.k = k
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, data: Data):
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization="sym",
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        if num_nodes < self.SPARSE_THRESHOLD:
            from numpy.linalg import eig, eigh

            eig_fn = eig if not self.is_undirected else eigh

            eig_vals, eig_vecs = eig_fn(L.todense())  # type: ignore
        else:
            from scipy.sparse.linalg import eigs, eigsh

            eig_fn = eigs if not self.is_undirected else eigsh

            eig_vals, eig_vecs = eig_fn(  # type: ignore
                L,
                k=self.k + 1,
                which="SR" if not self.is_undirected else "SA",
                return_eigenvectors=True,
                **self.kwargs,
            )

        sort_idx = eig_vals.argsort()
        eig_vecs = np.real(eig_vecs[:, sort_idx])

        data.EigVecs = torch.from_numpy(eig_vecs[:, 1 : self.k + 1])
        data.EigVals = torch.from_numpy(np.real(eig_vals[sort_idx][1 : self.k + 1]))

        # pad
        if data.EigVecs.shape[1] < self.k:
            data.EigVecs = torch.cat(
                [
                    data.EigVecs,
                    data.EigVecs.new_zeros(num_nodes, self.k - data.EigVecs.shape[1]),
                ],
                dim=1,
            )
            data.EigVals = torch.cat(
                [data.EigVals, data.EigVals.new_zeros(self.k - data.EigVecs.shape[1])],
                dim=0,
            )

        return data


class AugmentWithPartition:
    """
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/cluster.html#ClusterData
    """

    def __init__(self, num_parts, recursive=True):
        super(AugmentWithPartition, self).__init__()
        self.num_parts = num_parts
        self.recursive = recursive

    def __call__(self, graph: Data):
        row, index = sort_edge_index(
            graph.edge_index, num_nodes=graph.num_nodes, sort_by_row=True
        )
        indptr = index2ptr(row, size=graph.num_nodes)

        cluster = torch.ops.torch_sparse.partition(
            indptr.cpu(),
            index.cpu(),
            None,
            self.num_parts,
            self.recursive,
        ).to(graph.edge_index.device)

        graph.partition = cluster
        return graph


class AugmentWithDumbAttr:
    def __call__(self, graph: Data):
        graph.x = torch.ones(graph.num_nodes, 1, dtype=torch.float)
        graph.edge_attr = torch.ones(graph.num_edges, 1, dtype=torch.float)
        return graph


class RenameLabel:
    # dumb class to rename edge_label to y
    def __call__(self, graph: Data):
        graph.y = graph.edge_label.float()  # for BCE loss
        del graph.edge_label
        return graph


class AddLaplacianEigenvectorPE:
    """
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.AddLaplacianEigenvectorPE.html
    """

    # Number of nodes from which to use sparse eigenvector computation:
    SPARSE_THRESHOLD: int = 100

    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        **kwargs: Any,
    ) -> None:
        self.k = k
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def __call__(self, data: Data):
        assert data.edge_index is not None
        num_nodes = data.num_nodes
        assert num_nodes is not None

        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization="sym",
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        if num_nodes < self.SPARSE_THRESHOLD:
            from numpy.linalg import eig, eigh

            eig_fn = eig if not self.is_undirected else eigh

            eig_vals, eig_vecs = eig_fn(L.todense())  # type: ignore
        else:
            from scipy.sparse.linalg import eigs, eigsh

            eig_fn = eigs if not self.is_undirected else eigsh

            eig_vals, eig_vecs = eig_fn(  # type: ignore
                L,
                k=self.k + 1,
                which="SR" if not self.is_undirected else "SA",
                return_eigenvectors=True,
                **self.kwargs,
            )

        sort_idx = eig_vals.argsort()
        eig_vecs = np.real(eig_vecs[:, sort_idx])

        data.EigVecs = torch.from_numpy(eig_vecs[:, 1 : self.k + 1])
        data.EigVals = torch.from_numpy(np.real(eig_vals[sort_idx][1 : self.k + 1]))

        # pad
        if data.EigVecs.shape[1] < self.k:
            data.EigVecs = torch.cat(
                [
                    data.EigVecs,
                    data.EigVecs.new_zeros(num_nodes, self.k - data.EigVecs.shape[1]),
                ],
                dim=1,
            )
            data.EigVals = torch.cat(
                [data.EigVals, data.EigVals.new_zeros(self.k - data.EigVecs.shape[1])],
                dim=0,
            )

        return data


PRETRANSFORM_PRIORITY = {
    AugmentWithConformers: 99,
    PygWithConformers: 100,
    ToUndirected: 99,
    AddRemainingSelfLoops: 100,
    AddRandomWalkPE: 98,
    AddLaplacianEigenvectorPE: 98,
    AugmentWithPartition: 98,
    AugmentWithDumbAttr: 98,
    RenameLabel: 0,
}


def get_transform(args: Config):
    transform = []
    if transform:
        return Compose(transform)
    else:
        return None


def get_pretransform(args: Config, extra_pretransforms: Optional[List] = None):
    pretransform = []
    if extra_pretransforms is not None:
        pretransform = pretransform + extra_pretransforms

    if hasattr(args, "rwse_dim") or hasattr(args, "lap_dim"):
        if hasattr(args, "rwse_dim"):
            pretransform.append(AddRandomWalkPE(20, "pestat_RWSE"))
        if hasattr(args, "lap_dim"):
            pretransform.append(AddLaplacianEigenvectorPE(4, is_undirected=True))
    if hasattr(args, "n_conformers"):
        pretransform.append(AugmentWithConformers(num_conformers=args.n_conformers))

    if hasattr(args, "n_conformers_pyg"):
        pretransform.append(PygWithConformers(num_conformers=args.n_conformers_pyg))

    if pretransform:
        pretransform = sorted(
            pretransform, key=lambda p: PRETRANSFORM_PRIORITY[type(p)], reverse=True
        )
        return Compose(pretransform)
    else:
        return None


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def get_target_metric(args):
    if args.dataset.lower() == "zinc":
        target_metric = "mae"
        criterion = torch.nn.L1Loss()
    elif args.dataset.lower() in ["esol", "bace", "freesolv", "lipo"]:
        target_metric = "rmse"
        criterion = RMSELoss()
        # criterion = partial(mean_squared_error, squared=False)
    elif args.dataset.lower() == "kraken":
        target_metric = "mae_kraken"
        criterion = torch.nn.L1Loss()
    elif args.dataset.lower() == "drugs":
        target_metric = "mae_drugs"
        criterion = torch.nn.L1Loss()
    elif args.dataset.lower().startswith("drugs_"):
        target_metric = "mae"
        criterion = torch.nn.L1Loss()
    elif args.dataset.lower().startswith("kraken_"):
        target_metric = "mae"
        criterion = torch.nn.L1Loss()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    return target_metric, criterion
