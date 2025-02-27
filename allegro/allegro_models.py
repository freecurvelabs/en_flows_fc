import torch
import torch.nn as nn
import numpy as np

from flows.utils import remove_mean, remove_mean_with_mask

import inspect
from importlib import import_module
from typing import Optional

from nequip import data
from nequip.data import (
    DataLoader,
    PartialSampler,
    AtomicData,
    AtomicDataDict,
    AtomicDataset,
)
from nequip.data.transforms import TypeMapper
from nequip.nn import GraphModuleMixin, GraphModel
from nequip.utils import (
    load_callable,
    instantiate,
    get_w_prefix,
    dtype_from_name,
    torch_default_dtype,
    Config,
)
from nequip.utils.config import _GLOBAL_ALL_ASKED_FOR_KEYS

class ALLEGRO_dynamics(nn.Module):
    def __init__(self, args, n_particles, n_dimension, mode = 'allegro_dynamics', condition_time = True, device='cpu'):
        super().__init__()
        
        config = Config.from_file(args.config)
        #for flag in (   # for substituion of config parameters from command line arguments
        #    "model_debug_mode",
        #):
        #    config[flag] = getattr(args, flag) or config[flag]
        
        self.device = device
        
        seed = 2025
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.dataset_rng = torch.Generator()
        self.dataset_rng.manual_seed(seed)
        
        self.dataset_allegro = allegro_dataset_from_config(config)
        
        total_n = len(self.dataset_allegro)
        n_train = total_n
        
        idcs = torch.arange(total_n)
        self.train_idcs = idcs[:n_train]
        
        self.dataset_train = self.dataset_allegro.index_select(self.train_idcs)
        
        
        # based on recommendations from
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation
        self.exclude_keys = []
        self.dataloader_num_workers = 1
        self.max_epochs = 100
        self.shuffle = True
        self.n_train_per_epoch = 1000
        self.batch_size = 100
        
        dl_kwargs = dict(
            exclude_keys=self.exclude_keys,
            num_workers=self.dataloader_num_workers,
            # keep stuff around in memory
            persistent_workers=(
                self.dataloader_num_workers > 0 and self.max_epochs > 1
            ),
            # PyTorch recommends this for GPU since it makes copies much faster
            pin_memory=(self.device != torch.device("cpu")),
            # avoid getting stuck
            timeout=(10 if self.dataloader_num_workers > 0 else 0),
            # use the right randomness
            generator=self.dataset_rng,
        )
        
        self.dl_train_sampler = PartialSampler(
            data_source=self.dataset_train,
            # training should shuffle (if enabled)
            shuffle=self.shuffle,
            # if n_train_per_epoch is None (default), it's set to len(self.dataset_train) == n_train
            # i.e. use all `n_train` frames each epoch
            num_samples_per_epoch=self.n_train_per_epoch,
            generator=self.dataset_rng,
        )
        
        self.dl_train = DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            sampler=self.dl_train_sampler,
            **dl_kwargs,
        )
        
        self.mode = mode
        if mode == 'allegro_dynamics':
            # can set number of edge features here ( set in_edge_nf to 2 if adding bonding info)
            self.allegro_nn = allegro_model_from_config(config = config, initialize = True, dataset= self.dataset_allegro, deploy = False )
        elif mode == 'nequip_dynamics':
            self.allegro_nn = allegro_model_from_config(config = config, initialize = True, dataset= self.dataset_allegro, deploy = False )

        #import torchinfo
        #torchinfo.summary(self.allegro_nn,input_size=(1,6))
        #exit(1)
         
        self.device = device
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()   # edges - pairs of atom indexes [[row][col]] 
        self._edges_dict = {}               # this is empty but if not empty _cast_edges2batch will use it 
        self.condition_time = condition_time

    def forward(self, t, xs ):

        #print(f"ALLEGRO_dynamics.forward() {t=}, {xs.shape=}")
        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)   # forming the tensor of graph edges
        #edges = [edges[0], edges[1]]
        edges = torch.stack( (edges[0],edges[1]), dim=1 ).T
        n_edges = edges.shape[1]
        n_edges_mol = n_edges//n_batch
       
        x = xs.view(n_batch*self._n_particles, self._n_dimension).clone()
        atom_types_mol = torch.ones(self._n_particles, 1).to(self.device)             # forming the tensor of graph verticies (nodes?)  - all ones here!  
        edge_len_scale_mol = torch.ones(n_edges, 1).to(self.device)  
        
        # IGOR_TMP modify node feature vector (distinguish oxygens):
        #h[0,0] = 8.0
        #h[1,0] = 4.0
        for i in range(self._n_particles):
            if i % 3 == 0:
                atom_types_mol[i] = 0
        atom_types = atom_types_mol.repeat(n_batch,1)
        
        for j in range(n_edges_mol):
            i0 = edges[0][j]
            i1 = edges[1][j]
            if (i0 < 3 and i1 < 3) or (i0 > 2 and i1 > 2):
                edge_len_scale_mol[j] = 3.0
        edge_len_scale = edge_len_scale_mol.repeat(n_batch,1)
        
        h = atom_types
        
        #for i in range(n_batch):
        #    h[i*3,0]   = 0
        #    h[i*3+3,0] = 0
        #     h[i*3,0]   = 8.0
        #     h[i*3+1,0] = 2.0
        #     h[i*3+2,0] = 1.0
        #    h[i*6+3,0] = 0.0
        
        data_input = {}
        data_input[AtomicDataDict.POSITIONS_KEY] = x
        #data_input[AtomicDataDict.ATOMIC_NUMBERS_KEY] = h
        data_input[AtomicDataDict.ATOM_TYPE_KEY] = atom_types
        data_input[AtomicDataDict.ATOM_TYPE_KEY] = data_input[AtomicDataDict.ATOM_TYPE_KEY].long()
        data_input[AtomicDataDict.EDGE_INDEX_KEY ] = edges
        data_input[AtomicDataDict.EDGE_INDEX_KEY ] = data_input[AtomicDataDict.EDGE_INDEX_KEY ].long()
        data_input['edge_len_scale'] = edge_len_scale
        
        if self.condition_time:
            h = h*t

        if self.mode == 'allegro_dynamics' or self.mode == 'nequip_dynamics':
            #edge_attr = torch.sum((x[edges[0]] - x[edges[1]])**2, dim=1, keepdim=True)
            #print(f"{edge_attr.shape=}")
            #
            # IGOR_TMP adding bonding attributes - not working so far
            #bonding_attr = torch.ones(n_batch*self._n_particles*(self._n_particles-1), 1).to(self.device)
            #bond_attr_mol = torch.ones(self._n_particles*(self._n_particles-1), 1).to(self.device)
            #for i in range(self._n_particles*(self._n_particles-1)):
            #    bond_attr_mol[i] = i
            #    if( (edges[0][i] < 3 and edges[1][i] < 3) or (edges[0][i] > 2 and edges[1][i] > 2) ):
            #        bond_attr_mol[i] = 0.0
            #bonding_attr = bond_attr_mol.repeat(n_batch,1)
                    
            #edge_attr = torch.cat((edge_attr, bonding_attr), dim = 1)
            
            #data_input = self.dataset_allegro
            #data_input = next(iter(self.dl_train))
            #data_input = data_input.to(self.device)
            #data_input = AtomicData.to_AtomicDataDict(data_input)
            
            data_out = self.allegro_nn(data_input)
            vel = data_out[AtomicDataDict.FORCE_KEY]
            #x_final = data_out[AtomicDataDict.POSITIONS_KEY]
            #vel = x_final - x

        vel = vel.view(n_batch, self._n_particles, self._n_dimension)  
        vel = remove_mean(vel)
        vel = vel * 0.0001
        #print(f"ALLEGRO_dynamics.forward() {vel.shape=}")
        #print(f"ALLEGRO_dynamics.forward() {t=} {vel=}")
        return vel

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(self.device)
            cols_total = torch.cat(cols_total).to(self.device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]


def allegro_model_from_config(
    config: Config,
    initialize: bool = False,
    dataset: Optional[AtomicDataset] = None,
    deploy: bool = False,
) -> GraphModuleMixin:
    """Build NEQUIP or ALLEGRO model based on `config`.

    Model builders (`model_builders`) can have arguments:
     - ``config``: the config. Always present.
     - ``model``: the model produced by the previous builder. Cannot be requested by the first builder, must be requested by subsequent ones.
     - ``initialize``: whether to initialize the model
     - ``dataset``: if ``initialize`` is True, the dataset
     - ``deploy``: whether the model object is for deployment / inference

    Note that this function temporarily sets ``torch.set_default_dtype()`` and as such is not thread safe.

    Args:
        config
        initialize (bool): whether ``model_builders`` should be instructed to initialize the model
        dataset: dataset for initializers if ``initialize`` is True.
        deploy (bool): whether ``model_builders`` should be told the model is for deployment / inference

    Returns:
        The build model.
    """
    if isinstance(config, dict):
        config = Config.from_dict(config)
    # Pre-process config
    type_mapper = None
    if dataset is not None:
        type_mapper = dataset.type_mapper
    else:
        try:
            type_mapper, _ = instantiate(TypeMapper, all_args=config)
        except RuntimeError:
            pass

    if type_mapper is not None:
        if "num_types" in config:
            assert (
                config["num_types"] == type_mapper.num_types
            ), "inconsistant config & dataset"
        if "type_names" in config:
            assert (
                config["type_names"] == type_mapper.type_names
            ), "inconsistant config & dataset"
        config["num_types"] = type_mapper.num_types
        config["type_names"] = type_mapper.type_names
        config["type_to_chemical_symbol"] = type_mapper.type_to_chemical_symbol
        # We added them, so they are by definition valid:
        _GLOBAL_ALL_ASKED_FOR_KEYS.update(
            {"num_types", "type_names", "type_to_chemical_symbol"}
        )

    default_dtype = torch.get_default_dtype()
    model_dtype: torch.dtype = dtype_from_name(config.get("model_dtype", default_dtype))
    config["model_dtype"] = str(model_dtype).lstrip("torch.")
    # confirm sanity
    assert default_dtype in (torch.float32, torch.float64)
    if default_dtype == torch.float32 and model_dtype == torch.float64:
        raise ValueError(
            "Overall default_dtype=float32, but model_dtype=float64 is a higher precision- change default_dtype to float64"
        )
    # temporarily set the default dtype
    start_graph_model_builders = None
    with torch_default_dtype(model_dtype):

        # Build
        builders = [
            load_callable(b, prefix="nequip.model")
            for b in config.get("model_builders", [])
        ]

        model = None

        for builder_i, builder in enumerate(builders):
            pnames = inspect.signature(builder).parameters
            params = {}
            if "graph_model" in pnames:
                # start graph_model builders, which happen later
                start_graph_model_builders = builder_i
                break
            if "initialize" in pnames:
                params["initialize"] = initialize
            if "deploy" in pnames:
                params["deploy"] = deploy
            if "config" in pnames:
                params["config"] = config
            if "dataset" in pnames:
                if "initialize" not in pnames:
                    raise ValueError(
                        "Cannot request dataset without requesting initialize"
                    )
                if (
                    initialize
                    and pnames["dataset"].default == inspect.Parameter.empty
                    and dataset is None
                ):
                    raise RuntimeError(
                        f"Builder {builder.__name__} requires the dataset, initialize is true, but no dataset was provided to `model_from_config`."
                    )
                params["dataset"] = dataset
            if "model" in pnames:
                if model is None:
                    raise RuntimeError(
                        f"Builder {builder.__name__} asked for the model as an input, but no previous builder has returned a model"
                    )
                params["model"] = model
            else:
                if model is not None:
                    raise RuntimeError(
                        f"All model_builders after the first one that returns a model must take the model as an argument; {builder.__name__} doesn't"
                    )
            model = builder(**params)
            if model is not None and not isinstance(model, GraphModuleMixin):
                raise TypeError(
                    f"Builder {builder.__name__} didn't return a GraphModuleMixin, got {type(model)} instead"
                )
    # reset to default dtype by context manager

    # Wrap the model up
    model = GraphModel(
        model,
        model_dtype=model_dtype,
        model_input_fields=config.get("model_input_fields", {}),
    )

    # Run GraphModel builders
    if start_graph_model_builders is not None:
        for builder in builders[start_graph_model_builders:]:
            pnames = inspect.signature(builder).parameters
            params = {}
            assert "graph_model" in pnames
            params["graph_model"] = model
            if "model" in pnames:
                raise ValueError(
                    f"Once any builder requests `graph_model` (first requested by {builders[start_graph_model_builders].__name__}), no builder can request `model`, but {builder.__name__} did"
                )
            if "initialize" in pnames:
                params["initialize"] = initialize
            if "deploy" in pnames:
                params["deploy"] = deploy
            if "config" in pnames:
                params["config"] = config
            if "dataset" in pnames:
                if "initialize" not in pnames:
                    raise ValueError(
                        "Cannot request dataset without requesting initialize"
                    )
                if (
                    initialize
                    and pnames["dataset"].default == inspect.Parameter.empty
                    and dataset is None
                ):
                    raise RuntimeError(
                        f"Builder {builder.__name__} requires the dataset, initialize is true, but no dataset was provided to `model_from_config`."
                    )
                params["dataset"] = dataset

            model = builder(**params)
            if not isinstance(model, GraphModel):
                raise TypeError(
                    f"Builder {builder.__name__} didn't return a GraphModel, got {type(model)} instead"
                )

    return model

def allegro_dataset_from_config(config, prefix: str = "dataset") -> AtomicDataset:
    """initialize database based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Examples see tests/data/test_dataset.py TestFromConfig
    and tests/datasets/test_simplest.py

    Args:

    config (dict, nequip.utils.Config): dict/object that store all the parameters
    prefix (str): Optional. The prefix of all dataset parameters

    Return:

    dataset (nequip.data.AtomicDataset)
    """

    config_dataset = config.get(prefix, None)
    if config_dataset is None:
        raise KeyError(f"Dataset with prefix `{prefix}` isn't present in this config!")

    if inspect.isclass(config_dataset):
        # user define class
        class_name = config_dataset
    else:
        try:
            module_name = ".".join(config_dataset.split(".")[:-1])
            class_name = ".".join(config_dataset.split(".")[-1:])
            class_name = getattr(import_module(module_name), class_name)
        except Exception:
            # ^ TODO: don't catch all Exception
            # default class defined in nequip.data or nequip.dataset
            dataset_name = config_dataset.lower()

            class_name = None
            for k, v in inspect.getmembers(data, inspect.isclass):
                if k.endswith("Dataset"):
                    if k.lower() == dataset_name:
                        class_name = v
                    if k[:-7].lower() == dataset_name:
                        class_name = v
                elif k.lower() == dataset_name:
                    class_name = v

    if class_name is None:
        raise NameError(f"dataset type {dataset_name} does not exists")

    # if dataset r_max is not found, use the universal r_max
    atomicdata_options_key = "AtomicData_options"
    prefixed_eff_key = f"{prefix}_{atomicdata_options_key}"
    config[prefixed_eff_key] = get_w_prefix(
        atomicdata_options_key, {}, prefix=prefix, arg_dicts=config
    )
    config[prefixed_eff_key]["r_max"] = get_w_prefix(
        "r_max",
        prefix=prefix,
        arg_dicts=[config[prefixed_eff_key], config],
    )

    # Build a TypeMapper from the config
    type_mapper, _ = instantiate(TypeMapper, prefix=prefix, optional_args=config)

    instance, _ = instantiate(
        class_name,
        prefix=prefix,
        positional_args={"type_mapper": type_mapper},
        optional_args=config,
    )

    return instance
