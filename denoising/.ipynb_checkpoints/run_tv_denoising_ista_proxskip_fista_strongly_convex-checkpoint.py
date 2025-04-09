from cil.framework import ImageGeometry
from cil.optimisation.operators import GradientOperator
from cil.optimisation.functions import LeastSquares, L2NormSquared, MixedL21Norm
from cil.optimisation.algorithms import FISTA, ISTA
from cil.utilities import dataexample, noise
import numpy as np
from ProxSkip import ProxSkip
from cil.optimisation.utilities import MetricsDiagnostics, RSE
import inspect
import zarr
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from utils_cil import AdjointOperator, IndicatorMixedL21Norm, DistanceFromPrimalSolution


data = dataexample.SHAPES.get()
ig = data.geometry
noisy_data_np = zarr.load("../data/denoising/noisy_shapes_seed_10_var_0.05.zarr")["noisy"]
noisy_data_cil = ig.allocate()
noisy_data_cil.fill(noisy_data_np)

# optimal solution below for huber rof with parameters below
# alpha = 0.65
# epsilon = 0.025

alpha = 0.55
epsilon = 0.01

tv_denoising_dual_zarr = zarr.load("results/tv_denoising_dual_strongly_convex_shapes_alpha_{}.zarr".format(alpha))
# print(tv_denoising_dual_zarr)

dual_cvxpy_objective = tv_denoising_dual_zarr["dual_objective"]
dual_cvxpy_denoised_np = tv_denoising_dual_zarr["denoised"]
dual_cvxpy_denoised_cil = ig.allocate()
dual_cvxpy_denoised_cil.fill(dual_cvxpy_denoised_np)


Grad = GradientOperator(ig)
Div = AdjointOperator(Grad) 
Ltmp = Div.PowerMethod(Div, max_iteration=1000, initial=None, tolerance = 1e-6,  return_all=False)**2
print(f"Lipschitz constant is {Ltmp}, close to 8")
L = 8.0

operator = Div
F = (LeastSquares(A = operator, b=-noisy_data_cil, c=0.5) - 0.5*noisy_data_cil.squared_norm()) + epsilon/(2*alpha)*L2NormSquared()
tmp = alpha*MixedL21Norm()
G = IndicatorMixedL21Norm(tmp)


cb_dual_denoised = DistanceFromPrimalSolution(reference_solution=dual_cvxpy_denoised_cil, 
                                                data=noisy_data_cil, 
                                                operator = operator, 
                                                metrics_dict={"rse_primal_cvxpy":RSE, "ssim":SSIM, "psnr":PSNR})


num_runs = 30
num_iterations = 5000
mu = epsilon/alpha
prob = np.sqrt(mu/L)

initial= Div.domain.allocate()

step_size = 1./L


for run in range(num_runs):

    print("FISTA: Start run {}\n".format(run))
    fista = FISTA(initial=initial, f=F, g=G, 
                max_iteration=num_iterations, 
                update_objective_interval=1, 
                step_size=step_size)
    fista.run(verbose=0, callback = [cb_dual_denoised])  
    fista_denoised_image = noisy_data_cil + Div.direct(fista.solution)
    print("FISTA: Finish run {}\n".format(run))

    dd_fista = zarr.open_group("results/denoising_runs_alpha_{}_epsilon_{}/fista_run_{}_its_{}.zarr".format(alpha, epsilon, run, num_iterations))
    dd_fista["solution"] = [fista.solution[0].array, fista.solution[1].array]
    dd_fista["dual_objective"] = fista.objective
    dd_fista["timing"] = fista.timing
    dd_fista["fista_denoised_image"] = fista_denoised_image.array
    dd_fista["rse_primal_cvxpy"] = fista.rse_primal_cvxpy
    dd_fista["psnr"] = fista.psnr
    dd_fista["ssim"] = fista.ssim

    print("ISTA: Start run {}\n".format(run))
    ista = ISTA(initial=initial, f=F, g=G, 
                max_iteration=num_iterations, 
                update_objective_interval=1, 
                step_size=step_size)
    ista.run(verbose=0, callback = [cb_dual_denoised]) 
    ista_denoised_image = noisy_data_cil + Div.direct(ista.solution)
    print("ISTA: Finish run {}\n".format(run))

    dd_ista = zarr.open_group("results/denoising_runs_alpha_{}_epsilon_{}/ista_run_{}_its_{}.zarr".format(alpha, epsilon, run, num_iterations))
    dd_ista["solution"] = [ista.solution[0].array, ista.solution[1].array]
    dd_ista["dual_objective"] = ista.objective
    dd_ista["timing"] = ista.timing
    dd_ista["ista_denoised_image"] = ista_denoised_image.array
    dd_ista["rse_primal_cvxpy"] = ista.rse_primal_cvxpy
    dd_ista["psnr"] = ista.psnr
    dd_ista["ssim"] = ista.ssim    
    
    print("ProxSkip: Start run {}, prob = {}\n".format(run, prob))
    proxskip = ProxSkip(initial = initial, f = F, step_size = step_size, g=G, 
                update_objective_interval = 1, prob=prob, seed=40,
                max_iteration = num_iterations) 
    proxskip.run(verbose=0, callback=[cb_dual_denoised]) 
    print("ProxSkip: Finish run {}, prob = {}\n".format(run, prob))
    proxskip_denoised_image = noisy_data_cil + Div.direct(proxskip.solution)
    
    dd_proxskip = zarr.open_group("results/denoising_runs_alpha_{}_epsilon_{}/proxskip_run_{}_prob_{:.5f}_its_{}.zarr".format(alpha, epsilon, run, prob, num_iterations))
    dd_proxskip["solution"] = [proxskip.solution[0].array, proxskip.solution[1].array]
    dd_proxskip["dual_objective"] = proxskip.objective
    dd_proxskip["timing"] = proxskip.timing
    dd_proxskip["ista_denoised_image"] = proxskip_denoised_image.array  
    dd_proxskip["rse_primal_cvxpy"] = proxskip.rse_primal_cvxpy
    dd_proxskip["psnr"] = proxskip.psnr
    dd_proxskip["ssim"] = proxskip.ssim 
    dd_proxskip["use_prox"] = proxskip.use_prox 
    dd_proxskip["no_use_prox"] = proxskip.no_use_prox 

