import hashlib
import json
import os
from ast import literal_eval
from typing import Any, Dict, List, Tuple, Union

import yaml
from multimethod import multimethod


def create_nested_dict(flat_dict):
    nested_dict = {}
    for compound_key, value in flat_dict.items():
        keys = compound_key.split(".")
        d = nested_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    return nested_dict


class Config(dict):
    def __getattr__(self, key: str) -> Any:
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    def load(self, fpath: str, *, recursive: bool = False) -> None:
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        fpaths = [fpath]
        if recursive:
            extension = os.path.splitext(fpath)[1]
            while os.path.dirname(fpath) != fpath:
                fpath = os.path.dirname(fpath)
                fpaths.append(os.path.join(fpath, "default" + extension))
        for fpath in reversed(fpaths):
            if os.path.exists(fpath):
                with open(fpath) as f:
                    self.update(yaml.safe_load(f))

    def reload(self, fpath: str, *, recursive: bool = False) -> None:
        self.clear()
        self.load(fpath, recursive=recursive)

    @multimethod
    def update(self, other: Dict) -> None:
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], Config):
                    self[key] = Config()
                self[key].update(value)
            else:
                self[key] = value

    @multimethod
    def update(self, opts: Union[List, Tuple]) -> None:
        index = 0
        while index < len(opts):
            opt = opts[index]
            if opt.startswith("--"):
                opt = opt[2:]
            if "=" in opt:
                key, value = opt.split("=", 1)
                index += 1
            else:
                key, value = opt, opts[index + 1]
                index += 2
            current = self
            subkeys = key.split(".")
            try:
                value = literal_eval(value)
            except:
                pass
            for subkey in subkeys[:-1]:
                current = current.setdefault(subkey, Config())
            current[subkeys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        configs = dict()
        for key, value in self.items():
            if isinstance(value, Config):
                value = value.to_dict()
            configs[key] = value
        return configs

    def hash(self) -> str:
        buffer = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(buffer.encode()).hexdigest()

    def __str__(self) -> str:
        texts = []
        for key, value in self.items():
            if isinstance(value, Config):
                seperator = "\n"
            else:
                seperator = " "
            text = key + ":" + seperator + str(value)
            lines = text.split("\n")
            for k, line in enumerate(lines[1:]):
                lines[k + 1] = (" " * 2) + line
            texts.extend(lines)
        return "\n".join(texts)


def args_canonize(args: Union[Config, Dict]):
    for k, v in args.items():
        if isinstance(v, Union[Config, Dict]):
            args[k] = args_canonize(v)
        if isinstance(v, str):
            if v.lower() == "true":
                args[k] = True
            elif v.lower() == "false":
                args[k] = False
            elif v.lower() == "none":
                args[k] = None
    return args


def args_unify(args: Config):
    if (
        hasattr(args, "scorer_model")
        and args.scorer_model is not None
        and hasattr(args, "sampler")
        and args.sampler is not None
    ):
        if isinstance(args.scorer_model.num_centroids, int):
            assert args.sampler.sample_k <= args.scorer_model.num_centroids
            args.scorer_model.num_centroids = [
                args.scorer_model.num_centroids
            ] * args.sampler.num_ensemble
        elif isinstance(args.scorer_model.num_centroids, str):
            num_centroids = eval(args.scorer_model.num_centroids)
            assert isinstance(num_centroids, list)
            assert args.sampler.sample_k <= min(num_centroids)
            args.scorer_model.num_centroids = sorted(num_centroids)
            args.sampler.num_ensemble = len(num_centroids)
        else:
            raise TypeError
    return args
