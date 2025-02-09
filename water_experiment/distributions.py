import os
import mdtraj as md
import shutil
import numpy as np
import pandas as pd
import subprocess
import torch
from flows.utils import \
    center_gravity_zero_gaussian_log_likelihood_with_mask, \
    standard_gaussian_log_likelihood_with_mask, \
    center_gravity_zero_gaussian_log_likelihood, \
    sample_center_gravity_zero_gaussian_with_mask, \
    sample_center_gravity_zero_gaussian, \
    sample_gaussian_with_mask,remove_mean
from .dataset import get_data_wat
from . import losses
    
from typing import Union, Optional, Sequence
from collections.abc import Sequence as _Sequence
import warnings

def _is_non_empty_sequence_of_integers(x):
    return (
        isinstance(x, _Sequence) and (len(x) > 0) and all(isinstance(y, int) for y in x)
    )

def _is_sequence_of_non_empty_sequences_of_integers(x):
    return (
        isinstance(x, _Sequence)
        and len(x) > 0
        and all(_is_non_empty_sequence_of_integers(y) for y in x)
    )

def _parse_dim(dim):
    if isinstance(dim, int):
        return [torch.Size([dim])]
    if _is_non_empty_sequence_of_integers(dim):
        return [torch.Size(dim)]
    elif _is_sequence_of_non_empty_sequences_of_integers(dim):
        return list(map(torch.Size, dim))
    else:
        raise ValueError(
            f"dim must be either:"
            f"\n\t- an integer"
            f"\n\t- a non-empty list of integers"
            f"\n\t- a list with len > 1 containing non-empty lists containing integers"
        )

class Energy(torch.nn.Module):
    """
    Base class for all energy models.

    It supports energies defined over:
        - simple vector states of shape [..., D]
        - tensor states of shape [..., D1, D2, ..., Dn]
        - states composed of multiple tensors (x1, x2, x3, ...)
          where each xi is of form [..., D1, D2, ...., Dn]

    Each input can have multiple batch dimensions,
    so a final state could have shape
        ([B1, B2, ..., Bn, D1, D2, ..., Dn],
         ...,
         [B1, B2, ..., Bn, D'1, ..., D'1n]).

    which would return an energy tensor with shape
        ([B1, B2, ..., Bn, 1]).

    Forces are computed for each input by default.
    Here the convention is followed, that forces will have
    the same shape as the input state.

    To define the state shape, the parameter `dim` has to
    be of the following form:
        - an integer, e.g. d = 5
            then each event is a simple vector state
            of shape [..., 5]
        - a non-empty list of integers, e.g. d = [3, 6, 7]
            then each event is a tensor state of shape [..., 3, 6, 7]
        - a list of len > 1 containing non-empty integer lists,
            e.g. d = [[1, 3], [5, 3, 6]]. Then each event is
            a tuple of tensors of shape ([..., 1, 3], [..., 5, 3, 6])

    Parameters:
    -----------
    dim: Union[int, Sequence[int], Sequence[Sequence[int]]]
        The event shape of the states for which energies/forces ar computed.

    """

    def __init__(self, dim: Union[int, Sequence[int], Sequence[Sequence[int]]], **kwargs):

        super().__init__(**kwargs)
        self._event_shapes = _parse_dim(dim)

    @property
    def dim(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore there exists no coherent way to define the dimension of an event."
                "Consider using Energy.event_shapes instead."
            )
        elif len(self._event_shapes[0]) > 1:
            warnings.warn(
                "This Energy instance is defined on multidimensional events. "
                "Therefore, its Energy.dim is distributed over multiple tensor dimensions. "
                "Consider using Energy.event_shape instead.",
                UserWarning,
            )
        return int(torch.prod(torch.tensor(self.event_shape, dtype=int)))

    @property
    def event_shape(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore therefore there exists no single event shape."
                "Consider using Energy.event_shapes instead."
            )
        return self._event_shapes[0]

    @property
    def event_shapes(self):
        return self._event_shapes

    def _energy(self, *xs, **kwargs):
        raise NotImplementedError()

    def energy(self, *xs, temperature=1.0, **kwargs):
        assert len(xs) == len(
            self._event_shapes
        ), f"Expected {len(self._event_shapes)} arguments but only received {len(xs)}"
        batch_shape = xs[0].shape[: -len(self._event_shapes[0])]
        for i, (x, s) in enumerate(zip(xs, self._event_shapes)):
            assert x.shape[: -len(s)] == batch_shape, (
                f"Inconsistent batch shapes."
                f"Input at index {i} has batch shape {x.shape[:-len(s)]}"
                f"however input at index 0 has batch shape {batch_shape}."
            )
            assert (
                x.shape[-len(s) :] == s
            ), f"Input at index {i} as wrong shape {x.shape[-len(s):]} instead of {s}"
        return self._energy(*xs, **kwargs) / temperature

    def force(
        self,
        *xs: Sequence[torch.Tensor],
        temperature: float = 1.0,
        ignore_indices: Optional[Sequence[int]] = None,
        no_grad: Union[bool, Sequence[int]] = False,
        **kwargs,
    ):
        """
        Computes forces with respect to the input tensors.

        If states are tuples of tensors, it returns a tuple of forces for each input tensor.
        If states are simple tensors / vectors it returns a single forces.

        Depending on the context it might be unnecessary to compute all input forces.
        For this case `ignore_indices` denotes those input tensors for which no forces.
        are to be computed.

        E.g. by setting `ignore_indices = [1]` the result of `energy.force(x, y, z)`
        will be `(fx, None, fz)`.

        Furthermore, the forces will allow for taking high-order gradients by default.
        If this is unwanted, e.g. to save memory it can be turned off by setting `no_grad=True`.
        If higher-order gradients should be ignored for only a subset of inputs it can
        be specified by passing a list of ignore indices to `no_grad`.

        E.g. by setting `no_grad = [1]` the result of `energy.force(x, y, z)`
        will be `(fx, fy, fz)`, where `fx` and `fz` allow for taking higher order gradients
        and `fy` will not.

        Parameters:
        -----------
        xs: *torch.Tensor
            Input tensor(s)
        temperature: float
            Temperature at which to compute forces
        ignore_indices: Sequence[int]
            Which inputs should be skipped in the force computation
        no_grad: Union[bool, Sequence[int]]
            Either specifies whether higher-order gradients should be computed at all,
            or specifies which inputs to leave out when computing higher-order gradients.
        """
        if ignore_indices is None:
            ignore_indices = []

        with torch.enable_grad():
            forces = []
            requires_grad_states = [x.requires_grad for x in xs]

            for i, x in enumerate(xs):
                if i not in ignore_indices:
                    x = x.requires_grad_(True)
                else:
                    x = x.requires_grad_(False)

            energy = self.energy(*xs, temperature=temperature, **kwargs)

            for i, x in enumerate(xs):
                if i not in ignore_indices:
                    if isinstance(no_grad, bool):
                        with_grad = not no_grad
                    else:
                        with_grad = i not in no_grad
                    force = -torch.autograd.grad(
                        energy.sum(), x, create_graph=with_grad,
                    )[0]
                    forces.append(force)
                    x.requires_grad_(requires_grad_states[i])
                else:
                    forces.append(None)

        forces = (*forces,)
        if len(self._event_shapes) == 1:
            forces = forces[0]
        return forces

class PositionFeaturePrior(torch.nn.Module):
    def __init__(self, n_dim, in_node_nf):
        super().__init__()
        self.n_dim = n_dim
        self.in_node_nf = in_node_nf

    def forward(self, z_x, z_h, node_mask=None):
        assert len(z_x.size()) == 3
        assert len(node_mask.size()) == 3
        assert node_mask.size()[:2] == z_x.size()[:2]

        assert (z_x * (1 - node_mask)).sum() < 1e-8 and \
               (z_h * (1 - node_mask)).sum() < 1e-8, \
               'These variables should be properly masked.'

        log_pz_x = center_gravity_zero_gaussian_log_likelihood_with_mask(
            z_x, node_mask
        )

        log_pz_h = standard_gaussian_log_likelihood_with_mask(
            z_h, node_mask
        )

        log_pz = log_pz_x + log_pz_h
        return log_pz

    def sample(self, n_samples, n_nodes, node_mask):
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dim), device=node_mask.device,
            node_mask=node_mask)
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)

        return z_x, z_h

class PositionPrior(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return center_gravity_zero_gaussian_log_likelihood(x)

    def sample(self, size, device):
        samples = sample_center_gravity_zero_gaussian(size, device)
        return samples

class _ArbalestEnergyWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ene_model):
        energy, force, *_ = ene_model.evaluate(input)
        ctx.save_for_backward(-force)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        neg_force, = ctx.saved_tensors
        grad_input = grad_output * neg_force
        grad_input = grad_input.view(grad_input.shape[0], -1,3)
        return grad_input, None
    
_evaluate_arbalest_energy = _ArbalestEnergyWrapper.apply

class ArbalestPrior(Energy):
    def __init__(self, system_struct_fname: str, conf_templ_fname: str, crd_dim : int, coords : np.ndarray, ctx):
        """Torch module to compute reduced energies (Log Probabilities) using Arbalest

        Args:
            system_struct_fname (str): Molecular system structure file (e.g. eth_gly_1.gro)
            conf_templ_fname    (str): Arbalest RERUN configuration file template ( e.g. water_arrow_rerun_conf_templ.xml )
        """
        if os.name == 'nt':
            #self._arbalest_path = os.path.join("C:\\","MYPROG","ARBALEST_TRUNK","x64","Bin","Release","Arbalest")
            self._arbalest_path = os.path.join("C:\\","MYPROG","ARBALEST_NN_GPU","x64","Bin","Release","Arbalest")
        else:
            #self._arbalest_path = os.path.join("/srv","data","permanent","arbalest","BIN","Arbalest-nnff_gpu_ik-Double-NNADJ-march-sandybridge.r3673")
            self._arbalest_path = os.path.join("/home","igor","MYPROG","ARBALEST","build","ARBALEST","ARBALEST")
        
        self._mdtraj_topology = md.load(system_struct_fname).topology    # Set mdtraj topology
        self.n_atoms = self._mdtraj_topology.n_atoms
        self._dim = self.n_atoms*3
        self._kt_to_kcal = 8.314462*298/4.184/1000   # the conversion factor from kT to kcal/mol at 298K ( should make dependent on T)
        self.conf_templ_fname = conf_templ_fname
        self.ene_component = ""   # if not empty compute Arbalest Energy Component  
        
        self._coords = coords.reshape(-1,crd_dim)
        self._crd_dim = crd_dim
        self._npt     = len(coords)
        self._ctx = ctx
        
        event_shape = (self.n_atoms*3,) 
        super().__init__(event_shape)
        
    def forward(self, x):
        ene = self._energy(x)
        #(ene, frc) = self.evaluate_ene_frc(x)
        return -ene
    #    return center_gravity_zero_gaussian_log_likelihood(x)

    def sample(self, size, device):
        (n_samples,n_pt,n_dim) = size
        permutation = np.random.permutation(self._npt)[:n_samples]
        #permutation = np.random.permutation(self._npt)[:size]
        s = torch.tensor(np.asarray(self._coords[permutation]),device=device)
        s = s.reshape(n_samples,-1,3)
        return s
    
    def evaluate(self, batch):
        unitcell_lengths=np.full((len(batch),3),3.2)
        unitcell_angles=np.full((len(batch),3),90.0)
    
        trajectory = md.Trajectory(
            xyz=batch.cpu().detach().numpy().reshape(-1, self.n_atoms, 3), 
            topology= self._mdtraj_topology,
            unitcell_lengths=unitcell_lengths,
            unitcell_angles=unitcell_angles
        )
        
        traj_fname = "traj.trr"
        trajectory.save(traj_fname)
        conf_fname = "rerun_conf.xml"
        output_dir = "Output_rerun"
        fout = open(conf_fname,"w")
        with open(self.conf_templ_fname,"r") as finp:
            for line in finp:
                line = line.replace("TRR_TRAJ_FNAME",traj_fname)
                fout.write(line)
        fout.close()
        
        if os.path.exists(output_dir): 
            shutil.rmtree(output_dir)
        
        # May be not quite working in Code Jupyter
        # subprocess.run([self._arbalest_path,"--config",conf_fname],stdout=False,stderr=False, shell=False, check=True)
        #
        # TODO: Reduce Arbalest diagnostics to console when run from command line - too much output
        # print([self._arbalest_path,"--config",conf_fname])
        
        subprocess.run([self._arbalest_path,"--config",conf_fname],shell=False, check=False, 
                       stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT )
        
        energies = np.zeros((len(batch),1),dtype=np.float32)
        for fn in os.listdir(output_dir):
            if not fn.endswith(".ene"): continue
            fene_path = os.path.join(output_dir,fn)
            df = pd.read_csv(fene_path,sep="\s+")
            if self.ene_component == "":
                energies_np = df["EnrgPot"].to_numpy()/self._kt_to_kcal
            else:
                energies_np = df[self.ene_component].to_numpy()/self._kt_to_kcal
            energies[:,0] = energies_np
        
        forces = np.zeros((len(batch),self.n_atoms*3),dtype=np.float32)
        for fn in os.listdir(output_dir):
            if not fn.endswith(".force"): continue
            ffrc_path = os.path.join(output_dir,fn)
            finp = open(ffrc_path,"r")
            for i in range(len(batch)):
                line = finp.readline()
                for j in range(self.n_atoms):
                    line = finp.readline()
                    tokens = line.split("\t")
                    forces[i][j*3]   = float(tokens[0])
                    forces[i][j*3+1] = float(tokens[1])
                    forces[i][j*3+2] = float(tokens[2])
            finp.close()
        
        ene_tensor    = torch.tensor(energies,dtype=torch.float32).to(self._ctx)
        forces_tensor = torch.tensor(forces/(4.184*self._kt_to_kcal),dtype=torch.float32).to(self._ctx)  
        
        #forces_tensor = forces_tensor.reshape(len(batch),-1,3)
                
        return (ene_tensor,forces_tensor) 
    
    def _energy(self, batch, no_grads=False):
        return _evaluate_arbalest_energy(batch,self)
    
    def energy(self, batch):
        (ene,_) = self.evaluate(batch)
        return ene
         
    #def force(self, batch, temperature=None):
    #    (ene,frc) = self.evaluate(self, batch)
    #    return frc
    
def get_prior(args, ctx):
    
    if args.prior == "wat2_arrow" or args.prior == "wat2_gaff" or args.prior == "wat5_arrow" or args.prior == "wat5_gaff":
        prior = get_dist_water(args.prior, ctx)
    elif args.prior == "harmonic":
        prior = PositionPrior()
    else:
        prior = PositionPrior()  # set up Harmonic prior
    
    return prior

def get_target(args, ctx):
    
    if args.data == "wat2_arrow" or args.data == "wat2_gaff" or args.data == "wat5_arrow" or args.data == "wat5_gaff":
        target = get_dist_water(args.data, ctx)
    elif args.prior == "harmonic":
        target = PositionPrior()
    else:
        target = PositionPrior()  # set up Harmonic target
    
    return target

def get_dist_water(dist_name,ctx):
    
    if dist_name == "wat2_arrow":
        print("get_dist_water()  wat2_arrow")
        coords = np.load(os.path.join("water_experiment","data","wat2_arrow" + ".npy"))
        n_dims = 18
        dist = ArbalestPrior(os.path.join("water_experiment","data","wat_2.gro"), 
                              os.path.join("water_experiment","data","wat_2_arrow_rerun_conf_templ_oo_R5_KL_100.xml"),
                        n_dims, coords,ctx) 
        
    elif dist_name == 'wat2_gaff':
        print("get_dist_water()  wat2_gaff")
        coords = np.load(os.path.join("water_experiment","data","wat2_gaff" + ".npy"))
        n_dims = 18
        dist = ArbalestPrior(os.path.join("water_experiment","data","wat_2.gro"), 
                              os.path.join("water_experiment","data","wat_2_gaff_rerun_conf_templ_oo_R5_KL_100.xml"),
                        n_dims, coords,ctx)
        
    elif dist_name == 'wat5_arrow':
        coords = np.load(os.path.join("water_experiment","data","wat5_arrow" + ".npy"))
        n_dims = 45
        dist = ArbalestPrior(os.path.join("water_experiment","data","wat_5.gro"), 
                              os.path.join("water_experiment","data","wat_5_arrow_rerun_conf_templ_oo_R5.xml"), # should these be with _KL_100 suffix ??
                        n_dims, coords,ctx)
        
    elif dist_name == 'wat5_gaff':
        coords = np.load(os.path.join("water_experiment","data","wat5_gaff" + ".npy"))
        n_dims = 45
        dist = ArbalestPrior(os.path.join("water_experiment","data","wat_5.gro"), 
                              os.path.join("water_experiment","data","wat_5_gaff_rerun_conf_templ_oo_R5.xml"),
                        n_dims, coords,ctx)
        
    else:
        dist = None
    
    return dist

kt_to_kcal = 8.314462*298/4.184/1000  # the conversion factor from kT to kcal/mol at 298K

def test_flow(args, flow, ctx, flow_2 = None):
    
    flow = flow.to(ctx)
    if( flow_2 is not None):
        flow_2 = flow_2.to(ctx)
    
    dist_gaff = None
    dist_arrow = None
    data_gaff = None
    data_arrow = None
    
    if( args.data == 'wat2_gaff' or args.data == 'wat2_arrow'):
        dist_gaff = get_dist_water("wat2_gaff",ctx)
        dist_arrow = get_dist_water("wat2_arrow",ctx)
        data_arrow,batch_iter = get_data_wat(args.n_data, "test", 1000, 6, "wat2_arrow")
        data_gaff,batch_iter = get_data_wat(args.n_data, "test", 1000, 6, "wat2_gaff")
    elif( args.data == 'wat5_gaff' or args.data == 'wat5_arrow'):     
        dist_gaff = get_dist_water("wat5_gaff",ctx)
        dist_arrow = get_dist_water("wat5_arrow",ctx)
        data_arrow,batch_iter = get_data_wat(args.n_data, "test", 1000, 15, "wat5_arrow")
        data_gaff,batch_iter = get_data_wat(args.n_data, "test", 1000, 15, "wat5_gaff")
        
    
    print(f" test_flow() { data_gaff.shape = } {data_arrow.shape = }")
    
    gaff_ene_gaff_trj   = dist_gaff.energy(data_gaff[:1000]).cpu().detach().numpy()*kt_to_kcal  # Converted to kcal/mol
    print(f" {np.average(gaff_ene_gaff_trj)=} kcal/mol")
    gaff_ene_arrow_trj  = dist_gaff.energy(data_arrow[:1000]).cpu().detach().numpy()*kt_to_kcal  # Converted to kcal/mol
    print(f" {np.average(gaff_ene_arrow_trj)=} kcal/mol")
    arrow_ene_arrow_trj = dist_arrow.energy(data_arrow[:1000]).cpu().detach().numpy()*kt_to_kcal  # Converted to kcal/mol
    print(f" {np.average(arrow_ene_arrow_trj)=} kcal/mol")
    arrow_ene_gaff_trj  = dist_arrow.energy(data_gaff[:1000]).cpu().detach().numpy()*kt_to_kcal  # Converted to kcal/mol
    print(f" {np.average(arrow_ene_gaff_trj)=} kcal/mol")
    
    data_gaff = data_gaff.to(ctx)
    data_arrow = data_arrow.to(ctx)
    
    if args.prior == 'harmonic' and (args.data == 'wat2_gaff' or args.data == 'wat5_gaff'):
        
        batch_size = 1000
        batch = data_gaff[:batch_size].view(batch_size, -1, 3)
        batch = remove_mean(batch)
      
        prior = PositionPrior()
        losses.compute_loss_and_nll(args, flow, prior, batch)
        
        size = tuple(batch.shape)
        
        normal_sample = prior.sample(size, ctx.device)
        normal_to_gaff_data = flow.reverse(normal_sample)
        
        gaff_ene_transformed_normal = dist_gaff.energy(normal_to_gaff_data).cpu().detach().numpy()*kt_to_kcal  
        print(f" {np.average(gaff_ene_transformed_normal)=} kcal/mol")
        
        if( flow_2 is not None):
            z, delta_logp, reg_term  = flow(batch)
            gaff_to_arrow_data = flow_2.reverse(z)
            arrow_ene_transformed_gaff_trj = dist_arrow.energy(gaff_to_arrow_data).cpu().detach().numpy()*kt_to_kcal  
            print(f" {np.average(arrow_ene_transformed_gaff_trj)=} kcal/mol")
        
        return
        
    elif args.prior == 'harmonic' and (args.data == 'wat2_arrow' or args.data == 'wat5_arrow'):
      
        batch_size = 1000
        batch = data_arrow[:batch_size].view(batch_size, -1, 3)
        batch = remove_mean(batch)
      
        prior = PositionPrior()
        losses.compute_loss_and_nll(args, flow, prior, batch)
        
        normal_sample = prior.sample(batch_size, ctx)
        
        normal_to_arrow_data = flow.reverse(normal_sample)
        arrow_ene_transformed_normal = dist_arrow.energy(normal_to_arrow_data).cpu().detach().numpy()*kt_to_kcal  
        print(f" {np.average(arrow_ene_transformed_normal)=} kcal/mol")
        
        if( flow_2 is not None):
            z, delta_logp, reg_term  = flow(batch)
            arrow_to_gaff_data = flow_2.reverse(z)
            gaff_ene_transformed_arrow_trj = dist_gaff.energy(arrow_to_gaff_data).cpu().detach().numpy()*kt_to_kcal  
            print(f" {np.average(gaff_ene_transformed_arrow_trj)=} kcal/mol")
            
        return
        
        
    (gaff_to_arrow_data,dlogp_gaff_to_arrow, _) = flow(data_gaff[:1000])
    arrow_ene_transformed_gaff_trj = dist_arrow.energy(gaff_to_arrow_data).cpu().detach().numpy()*kt_to_kcal  # Converted to kcal/mol
    del gaff_to_arrow_data
    print(f" {np.average(arrow_ene_transformed_gaff_trj)=} kcal/mol")

    #(arrow_to_gaff_data,dlogp_arrow_to_gaff) = flow.forward(data_arrow[:1000], inverse=True)
    arrow_to_gaff_data = flow.reverse(data_arrow[:1000])
    gaff_ene_transformed_arrow_trj = dist_gaff.energy(arrow_to_gaff_data).cpu().detach().numpy()*kt_to_kcal  # Converted to kcal/mol
    del arrow_to_gaff_data
    print(f" {np.average(gaff_ene_transformed_arrow_trj)=} kcal/mol")
        
    
    
    
    
    

