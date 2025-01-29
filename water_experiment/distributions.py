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
    sample_gaussian_with_mask


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

class ArbalestPrior(torch.nn.Module):
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
        super().__init__()
        
    def forward(self, x):
        (ene, frc) = self.evaluate_ene_frc(x)
        return ene
    #    return center_gravity_zero_gaussian_log_likelihood(x)

    def sample(self, size, device):
        permutation = np.random.permutation(self._npt)[:size]
        return torch.tensor(np.asarray(self._coords[permutation]),device=device)
    
    def evaluate_ene_frc(self, batch):
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
        
        return (ene_tensor,forces_tensor) 
    
def get_prior(args, ctx):
    
    if args.data == "wat2_gaff":
        coords_prior = np.load(os.path.join("water_experiment","data","wat2_arrow" + ".npy"))
        n_dims = 18
        prior = ArbalestPrior(os.path.join("water_experiment","data","wat_2.gro"), 
                              os.path.join("water_experiment","data","wat_2_arrow_rerun_conf_templ_oo_R5_KL_100.xml"),
                        n_dims, coords_prior,ctx) 
    elif args.data == 'wat2_arrow':
        coords_prior = np.load(os.path.join("water_experiment","data","wat2_gaff" + ".npy"))
        n_dims = 18
        prior = ArbalestPrior(os.path.join("water_experiment","data","wat_2.gro"), 
                              os.path.join("water_experiment","data","wat_2_gaff_rerun_conf_templ_oo_R5_KL_100.xml"),
                        n_dims, coords_prior,ctx)
    elif args.data == 'wat5_gaff':
        coords_prior = np.load(os.path.join("water_experiment","data","wat5_arrow" + ".npy"))
        n_dims = 45
        prior = ArbalestPrior(os.path.join("water_experiment","data","wat_5.gro"), 
                              os.path.join("water_experiment","data","wat_5_arrow_rerun_conf_templ_oo_R5.xml"), # should these be with _KL_100 suffix ??
                        n_dims, coords_prior,ctx)
    elif args.data == 'wat5_arrow':
        coords_prior = np.load(os.path.join("water_experiment","data","wat5_gaff" + ".npy"))
        n_dims = 45
        prior = ArbalestPrior(os.path.join("water_experiment","data","wat_5.gro"), 
                              os.path.join("water_experiment","data","wat_5_gaff_rerun_conf_templ_oo_R5.xml"),
                        n_dims, coords_prior,ctx)
    else:
        prior = PositionPrior()  # set up Harmonic prior
    
    return prior

