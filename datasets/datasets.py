import os
import pickle
from collections import defaultdict, namedtuple

import pandas as pd
import torch
from rdkit import Chem
from sklearn.utils import shuffle
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.datasets import ZINC, MoleculeNet
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from marcel.loaders.utils import mol_to_data_obj
from utils.datasets_utils import get_pretransform, get_transform

NUM_WORKERS = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

AttributedDataLoader = namedtuple(
    "AttributedDataLoader",
    [
        "loader",
        "std",
        "task",
    ],
)

class MyDrugsDataset(InMemoryDataset):
    descriptors = ["energy", "ip", "ea", "chi"]

    def __init__(
        self,
        root,
        max_num_conformers=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        self.num_molecules = 75099  # hard-coded for now

    @property
    def processed_file_names(self):
        return (
            "DrugsEnsemble_processed.pt"
            if self.max_num_conformers is None
            else f"DrugsEnsemble_processed_{self.max_num_conformers}.pt"
        )

    @property
    def raw_file_names(self):
        return "Drugs.zip"

    @property
    def num_conformers(self):
        return len(self)

    def process(self):
        quantities = self.descriptors

        mols = defaultdict(list)

        raw_dir = "datasets/drugs/raw"
        raw_zip_file = "datasets/drugs/raw/Drugs.zip"
        raw_sdf_file = "datasets/drugs/raw/Drugs.sdf"
        extract_zip(raw_zip_file, raw_dir)
        with Chem.SDMolSupplier(raw_sdf_file, removeHs=False) as suppl:
            for idx, mol in enumerate(tqdm(suppl)):
                id_ = mol.GetProp("ID")
                name = mol.GetProp("_Name")
                smiles = mol.GetProp("smiles")

                data = mol_to_data_obj(mol)
                data.name = name
                data.id = id_

                data.smiles = smiles
                data.y = []
                for quantity in quantities:
                    data.y.append(float(mol.GetProp(quantity)))
                data.y = torch.Tensor(data.y).unsqueeze(0)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                mols[name].append(data)

        label_file = raw_sdf_file.replace(".sdf", ".csv")
        labels = pd.read_csv(label_file)

        data_list = []

        for name, mol_list in tqdm(mols.items()):
            row = labels[labels["name"] == name]

            y = torch.Tensor([row[quantity].item() for quantity in quantities])

            if "drugs_energy" in self.raw_paths[0]:
                y = y[0]
            elif "drugs_ip" in self.raw_paths[0]:
                y = y[1]
            elif "drugs_ea" in self.raw_paths[0]:
                y = y[2]
            elif "drugs_chi" in self.raw_paths[0]:
                y = y[3]
            else:
                raise NotImplementedError("Unknown dataset")

            if self.max_num_conformers is not None:
                # sort energy and take the lowest energy conformers
                mol_list = sorted(
                    mol_list, key=lambda x: x.y[:, quantities.index("energy")].item()
                )
                mol_list = mol_list[: self.max_num_conformers]

            if len(mol_list) < self.max_num_conformers:
                repeats = (self.max_num_conformers // len(mol_list)) + 1
                mol_list = (mol_list * repeats)[: self.max_num_conformers]

            data_list.append(
                Data(
                    x=mol_list[0].x,
                    z=mol_list[0].x[:, 0],
                    edge_index=mol_list[0].edge_index,
                    edge_attr=mol_list[0].edge_attr,
                    pos=[mol.pos for mol in mol_list],
                    y=y.unsqueeze(0),
                    smiles=mol_list[0].smiles,
                )
            )

        self.save(data_list, self.processed_paths[0])

    def get_idx_split(self, train_ratio=0.8, valid_ratio=0.1, seed=123):
        molecule_ids = shuffle(range(self.num_molecules), random_state=seed)
        train_size = int(train_ratio * self.num_molecules)
        valid_size = int(valid_ratio * self.num_molecules)

        train_idx = torch.tensor(molecule_ids[:train_size])
        valid_idx = torch.tensor(molecule_ids[train_size : train_size + valid_size])
        test_idx = torch.tensor(molecule_ids[train_size + valid_size :])

        return train_idx, valid_idx, test_idx


class MyKrakenDataset(InMemoryDataset):
    descriptors = ["sterimol_B5", "sterimol_L", "sterimol_burB5", "sterimol_burL"]

    def __init__(
        self,
        root,
        max_num_conformers=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # self.data, self.slices, self.y = out
        self.num_molecules = 1552  # hard-coded for now

    @property
    def processed_file_names(self):
        return (
            "Kraken_processed.pt"
            if self.max_num_conformers is None
            else f"Kraken_{self.max_num_conformers}_processed.pt"
        )

    @property
    def raw_file_names(self):
        return "Kraken.zip"

    def process(self):
        data_list = []
        descriptors = self.descriptors

        raw_dir = "datasets/kraken/raw"
        raw_zip_file = "datasets/kraken/raw/Kraken.zip"
        raw_pickle_file = "datasets/kraken/raw/Kraken.pickle"
        extract_zip(raw_zip_file, raw_dir)
        with open(raw_pickle_file, "rb") as f:
            kraken = pickle.load(f)

        ligand_ids = list(kraken.keys())
        y = []

        max_conformers = 0
        for ligand_id in tqdm(ligand_ids):
            _, _, conformer_dict = kraken[ligand_id]
            conformer_ids = list(conformer_dict.keys())
            n_conformers = len(conformer_ids)
            if n_conformers > max_conformers:
                max_conformers = n_conformers

        if self.max_num_conformers == "all":
            self.max_num_conformers = max_conformers

        for ligand_id in tqdm(ligand_ids):
            pos = []

            smiles, boltz_avg_properties, conformer_dict = kraken[ligand_id]
            conformer_ids = list(conformer_dict.keys())

            if self.max_num_conformers is not None:
                # sort conformers by boltzmann weight and take the lowest energy conformers
                conformer_ids = sorted(
                    conformer_ids, key=lambda x: conformer_dict[x][1], reverse=True
                )
                conformer_ids = conformer_ids[: self.max_num_conformers]

            if len(conformer_ids) < self.max_num_conformers:
                repeats = (self.max_num_conformers // len(conformer_ids)) + 1
                conformer_ids = (conformer_ids * repeats)[: self.max_num_conformers]

            for conformer_id in conformer_ids:
                mol_sdf, boltz_weight, conformer_properties = conformer_dict[
                    conformer_id
                ]
                mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False)

                data = mol_to_data_obj(mol)
                pos.append(data.pos)

            y = torch.tensor(
                [boltz_avg_properties[descriptor] for descriptor in descriptors]
            )

            if "_b5" in self.raw_paths[0]:
                y = y[0]
            elif "_l" in self.raw_paths[0]:
                y = y[1]
            elif "_burb5" in self.raw_paths[0]:
                y = y[2]
            elif "_burl" in self.raw_paths[0]:
                y = y[3]

            data_list.append(
                Data(
                    x=data.x,
                    z=data.x[:, 0],
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    pos=pos,
                    y=y.unsqueeze(0),
                    smiles=smiles,
                )
            )

        self.save(data_list, self.processed_paths[0])

    def get_idx_split(self, train_ratio=0.8, valid_ratio=0.1, seed=123):
        molecule_ids = shuffle(range(self.num_molecules), random_state=seed)
        train_size = int(train_ratio * self.num_molecules)
        valid_size = int(valid_ratio * self.num_molecules)

        train_idx = torch.tensor(molecule_ids[:train_size])
        valid_idx = torch.tensor(molecule_ids[train_size : train_size + valid_size])
        test_idx = torch.tensor(molecule_ids[train_size + valid_size :])

        return train_idx, valid_idx, test_idx

def get_data(args):
    task = "graph"

    if args.dataset.lower() == "zinc":
        train_set, val_set, test_set, std = get_zinc(args)
    elif args.dataset.lower() in ["esol", "freesolv", "lipo", "bace"]:
        train_set, val_set, test_set, std = get_moleculenet(args)
    elif args.dataset.lower().startswith("kraken"):
        train_set, val_set, test_set, std = get_kraken(args)
    elif args.dataset.lower().startswith("drugs"):
        train_set, val_set, test_set, std = get_drugs(args)
    else:
        raise NotImplementedError

    trn_loader = AttributedDataLoader(
        loader=PyGDataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=not args.debug,
            num_workers=NUM_WORKERS,
            drop_last=True,
            follow_batch=(
                ["x", "z"]
                if args.dataset.lower() in ["esol", "freesolv", "lipo", "bace"]
                or args.dataset.lower().startswith("kraken")
                or args.dataset.lower().startswith("drugs")
                else None
            ),
        ),
        std=std,
        task=task,
    )
    val_loader = AttributedDataLoader(
        loader=PyGDataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            follow_batch=(
                ["x", "z"]
                if args.dataset.lower() in ["esol", "freesolv", "lipo", "bace"]
                or args.dataset.lower().startswith("kraken")
                or args.dataset.lower().startswith("drugs")
                else None
            ),
        ),
        std=std,
        task=task,
    )
    tst_loader = AttributedDataLoader(
        loader=PyGDataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            follow_batch=(
                ["x", "z"]
                if args.dataset.lower() in ["esol", "freesolv", "lipo", "bace"]
                or args.dataset.lower().startswith("kraken")
                or args.dataset.lower().startswith("drugs")
                else None
            ),
        ),
        std=std,
        task=task,
    )

    return trn_loader, val_loader, tst_loader, task


def get_drugs(args):
    dataset_name = args.dataset.lower()
    datapath = os.path.join(args.data_path, dataset_name)

    drugs_dataset = MyDrugsDataset(root=datapath, max_num_conformers=args.n_conformers)

    train_idx, val_idx, test_idx = drugs_dataset.get_idx_split(
        train_ratio=0.8, valid_ratio=0.1, seed=123
    )

    train = drugs_dataset[train_idx]
    val = drugs_dataset[val_idx]
    test = drugs_dataset[test_idx]
    stds = train.y.std(dim=0, keepdim=True)
    mean = train.y.mean(dim=0, keepdim=True)

    return train, val, test, (mean, stds)


def get_kraken(args):
    dataset_name = args.dataset.lower()
    datapath = os.path.join(args.data_path, dataset_name)

    kraken_dataset = MyKrakenDataset(
        root=datapath, max_num_conformers=args.n_conformers
    )

    train_idx, val_idx, test_idx = kraken_dataset.get_idx_split(
        train_ratio=0.8, valid_ratio=0.1, seed=123
    )

    train = kraken_dataset[train_idx]
    val = kraken_dataset[val_idx]
    test = kraken_dataset[test_idx]
    stds = train.y.std(dim=0, keepdim=True)
    mean = train.y.mean(dim=0, keepdim=True)

    return train, val, test, (mean, stds)


def get_moleculenet(args):
    if args.dataset.lower() == "esol":
        dataset_name = "ESOL"
    elif args.dataset.lower() == "freesolv":
        dataset_name = "FreeSolv"
    elif args.dataset.lower() == "lipo":
        dataset_name = "Lipo"
    elif args.dataset.lower() == "bace":
        dataset_name = "BACE"
    else:
        raise NotImplementedError

    datapath = os.path.join(args.data_path, dataset_name)

    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    pre_transforms = get_pretransform(args)

    moleculenet_ds = MoleculeNet(
        root=datapath, name=dataset_name, pre_transform=pre_transforms
    )

    moleculenet_ds_clean = moleculenet_ds

    train_smiles = pd.read_csv(
        f"./datasets/splits/{dataset_name.lower()}/train_smiles.csv"
    )["smiles"]
    val_smiles = pd.read_csv(
        f"./datasets/splits/{dataset_name.lower()}/val_smiles.csv"
    )["smiles"]
    test_smiles = pd.read_csv(
        f"./datasets/splits/{dataset_name.lower()}/test_smiles.csv"
    )["smiles"]

    train_idx = [
        i
        for i, entry in enumerate(moleculenet_ds_clean)
        if train_smiles.isin([entry.smiles]).any()
    ]
    val_idx = [
        i
        for i, entry in enumerate(moleculenet_ds_clean)
        if val_smiles.isin([entry.smiles]).any()
    ]
    test_idx = [
        i
        for i, entry in enumerate(moleculenet_ds_clean)
        if test_smiles.isin([entry.smiles]).any()
    ]

    if args.dataset.lower() == "lipo":
        train_set = [moleculenet_ds_clean[i] for i in train_idx]
        val_set = [moleculenet_ds_clean[i] for i in val_idx]
        test_set = [moleculenet_ds_clean[i] for i in test_idx]
    else:
        train_set = moleculenet_ds[train_idx]
        val_set = moleculenet_ds[val_idx]
        test_set = moleculenet_ds[test_idx]

    return train_set, val_set, test_set, None


def get_additional_path(args):
    extra_path = ""
    if hasattr(args, "n_conformers"):
        extra_path += f"n_conformers_{args.n_conformers}_"
    if hasattr(args, "rwse_dim"):
        extra_path += "rwse_"
    if hasattr(args, "lap_dim"):
        extra_path += "lap_"
    return extra_path if len(extra_path) else None


def get_zinc(args):
    datapath = os.path.join(args.data_path, "ZINC")

    extra_path = get_additional_path(args)
    if extra_path is not None:
        datapath = os.path.join(datapath, extra_path)

    pre_transform = get_pretransform(args)
    transform = get_transform(args)

    train_set = ZINC(
        datapath,
        split="train",
        subset=True,
        transform=transform,
        pre_transform=pre_transform,
    )

    val_set = ZINC(
        datapath,
        split="val",
        subset=True,
        transform=transform,
        pre_transform=pre_transform,
    )

    test_set = ZINC(
        datapath,
        split="test",
        subset=True,
        transform=transform,
        pre_transform=pre_transform,
    )

    for sp in [train_set, val_set, test_set]:
        sp.data.x = sp.data.x.squeeze()
        sp.data.edge_attr = sp.data.edge_attr.squeeze()
        sp.data.y = sp.data.y[:, None]

    return train_set, val_set, test_set, None
