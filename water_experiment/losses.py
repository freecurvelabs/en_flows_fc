import torch
import numpy as np

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


# class UnnormalizedPrior(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self._dim = dim
#         # self._n_particles = n_particles
#         # self._spacial_dims = dim // n_particles
#
#     def forward(self, x):
#         assert len(x.size()) == 3
#         x = self._remove_mean(x)
#         log_px = sum_except_batch(-0.5 * x.pow(2).sum(dim=-1, keepdim=True))
#         return log_px
#
#     # def sample(self, n_samples, temperature=1.):
#     #     x = torch.Tensor(n_samples, self._n_particles,
#     #                      self._spacial_dims).normal_()
#     #     return self._remove_mean(x)
#
#     def _remove_mean(self, x):
#         x = x - torch.mean(x, dim=1, keepdim=True)
#         return x


def  compute_loss_and_nll(args, flow, prior, batch):
    # if args.ode_regularization > 0:
    #     z, delta_logp, reg_frob, reg_dx2 = flow(batch)
    #     z = z.view(z.size(0), 8)
    #     nll = (prior(z).view(-1) + delta_logp.view(-1)).mean()
    #     reg_term = (reg_frob.mean() + reg_dx2.mean())
    #     loss = nll + args.ode_regularization * reg_term
    # else:
    #print(f" compute_loss_and_nll() {batch.shape=}")
    
    z, delta_logp, reg_term = flow(batch)
    #z_max = z.max()
    #print(f"{z_max=}")
    #print(f"{z_max.requires_grad=}")
    
    #_max.retain_grad()
    #z_max.backward(retain_graph=True)
    #print(f"{z_max.grad=}")
    
    #x = batch.detach().cpu().numpy().reshape(batch.shape[0],-1,3) * 10.0
    #x_tr = z.detach().cpu().numpy().reshape(batch.shape[0],-1,3) * 10.0
    
    #np.set_printoptions(precision=4)
    #print(f"compute_loss_and_nll() \n x= {x}")
    #print(f" x_tr= \n {x_tr}")
    
    log_pz = prior(z)
    
    log_px = (log_pz + delta_logp.view(-1)).mean()  
    #  
    nll = -log_px

    print(f"{-log_pz.mean()=:.2f} {-delta_logp.mean()=:.2f} ")

    mean_abs_z = torch.mean(torch.abs(z)).item()
    # print(f"mean(abs(z)): {mean_abs_z:.2f}")

    #reg_term = torch.tensor([0.])
    reg_term = reg_term.mean()  # Average over batch.
    loss = nll

    return loss, nll, reg_term, mean_abs_z

def compute_kll_loss(args, flow, prior, target, n_samples, ctx, temperature=1.0):
    # if args.ode_regularization > 0:
    #     z, delta_logp, reg_frob, reg_dx2 = flow(batch)
    #     z = z.view(z.size(0), 8)
    #     nll = (prior(z).view(-1) + delta_logp.view(-1)).mean()
    #     reg_term = (reg_frob.mean() + reg_dx2.mean())
    #     loss = nll + args.ode_regularization * reg_term
    # else:
    #print(f" compute_loss_and_nll() {batch.shape=}")
    
    reg_term = torch.tensor([0.])
    
    z = prior.sample(n_samples, ctx.device)
    x, dlogp, reg_term = flow.inverse(z)
    kll = target._energy(x).mean()
    dlogp_avg = dlogp.mean()  
    #kll_tot = kll - dlogp_avg  # the sign of dlogp  is not clear
    kll_tot = kll       # IGOR_TMP removed dlogp_avg from the penalty
    
    print(f"{kll=:.2f} {dlogp_avg=:.2f} ")

    return kll_tot, kll, dlogp_avg, reg_term  


def compute_loss_and_nll_kerneldynamics(args, flow, prior, batch, n_particles, n_dims):
    bs = batch.size(0)
    z, delta_logp = flow(batch.view(bs, -1))
    z = z.view(bs, n_particles, n_dims)
    nll = -(prior(z).view(-1) - delta_logp.view(-1)).mean()
    loss = nll
    reg_term, mean_abs_z = torch.tensor([0.]), 0
    return loss, nll, reg_term.to(z.device), mean_abs_z
