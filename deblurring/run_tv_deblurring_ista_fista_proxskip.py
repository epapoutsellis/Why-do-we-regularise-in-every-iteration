from cil.framework import ImageGeometry
from cil.optimisation.operators import LinearOperator, GradientOperator, BlurringOperator
from cil.optimisation.functions import Function
from cil.optimisation.functions import MixedL21Norm, LeastSquares, L2NormSquared, L1Norm, ZeroFunction
from cil.optimisation.algorithms import FISTA, ISTA
from cil.utilities import dataexample, noise
from cil.utilities.display import show2D
import numpy as np
import matplotlib.pyplot as plt
from cil.optimisation.algorithms import ProxSkip
from cil.optimisation.utilities import MetricsDiagnostics, StatisticsDiagnostics, AlgorithmDiagnostics, RSE
from utils_cil import TimeStoppingCriterion
from TotalVariation import TotalVariationNew
import zarr
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['font.size'] = 20

class StoppingCriterion(AlgorithmDiagnostics):

    def __init__(self, epsilon):

        self.epsilon = epsilon
        super(StoppingCriterion, self).__init__(verbose=0)

        self.should_stop = False
        
    def _should_stop(self):

        return self.should_stop       
            
    def __call__(self, algo):

        if algo.iteration==0:
            algo.should_stop = self._should_stop  
        
        if (algo.rse[-1]<=self.epsilon):
            self.should_stop = True                 
            print("Stop at {} time {} ".format(algo.iteration, np.sum(algo.timing)))



# # Load data
data = dataexample.SHAPES.get()
ig = data.geometry


# Parameters for point spread function PSF (size and std)
ks          = 10; 
ksigma      = 5;

# Create 1D PSF and 2D as outer product, then normalise.
w           = np.exp(-np.arange(-(ks-1)/2,(ks-1)/2+1)**2/(2*ksigma**2))
w.shape     = (ks,1)
PSF         = w*np.transpose(w)
PSF         = PSF/(PSF**2).sum()
PSF         = PSF/PSF.sum()

# Create blurring operator and apply to clean image to produce blurred and display.
BOP = BlurringOperator(PSF,ig)


blurred_noisy = zarr.load("results/noisy_blurry_shapes_var_0.001_seed_10.zarr")


blurred_noisy_cil = ig.allocate()
blurred_noisy_cil.fill(blurred_noisy["blurred_noisy"])

alpha = 0.025

# compare vs fista optimal
# fista_optimal_info = zarr.load("results_deblurring/fista_optimal_shapes_tv_alpha_0.025_warm_500.zarr")
# fista_optimal_np = fista_optimal_info["solution"]
# fista_optimal_objective = fista_optimal_info["objective"]
# fista_optimal_cil = ig.allocate()
# fista_optimal_cil.fill(fista_optimal_np)

# compare vs pdhg precond
pdhg_precond_optimal_info = zarr.load("results_deblurring/pdhg_optimal_precond_shapes_tv_alpha_{}.zarr".format(alpha))
pdhg_precond_optimal_np = pdhg_precond_optimal_info["solution"]
pdhg_precond_optimal_objective = pdhg_precond_optimal_info["objective"]

pdhg_precond_optimal = ig.allocate()
pdhg_precond_optimal.fill(pdhg_precond_optimal_np)


G10 = alpha * TotalVariationNew(max_iteration=10, warm_start=True)
G100 = alpha * TotalVariationNew(max_iteration=100, warm_start=True)
G50 = alpha * TotalVariationNew(max_iteration=50, warm_start=True)

F = LeastSquares(A=BOP, b=blurred_noisy_cil, c=0.5)

G_dict = {10:G10, 50:G50, 100:G100}

def NRSE(x, y, **kwargs):
    """
     root squared error between two numpy arrays
    """
    return np.sqrt(np.sum(np.abs(x - y)**2))/np.sqrt(np.sum(x**2))

SSIM_dr = lambda x, y: SSIM(x, y, data_range=x.max()-x.min())
PSNR_dr = lambda x, y: PSNR(x, y, data_range=x.max()-x.min())


cb1 = MetricsDiagnostics(reference_image=pdhg_precond_optimal, 
                         metrics_dict={"rse":NRSE, "ssim":SSIM_dr, "psnr":PSNR_dr})  


num_runs = 10
step_size = 1.
num_iterations = 3000
initial = ig.allocate()

probs = [0.05, 0.1, 0.2, 0.5, 1.0]

for run in range(num_runs):

    for inner_its, G in G_dict.items():

        print("FISTA-{}: start run {}".format(inner_its, run))
        cb2 = StoppingCriterion(epsilon=0.99e-5)
        fista = FISTA(initial=initial, f=F, g=G, max_iteration=num_iterations, update_objective_interval=1, step_size=step_size)
        fista.run(verbose=0, callback=[cb1, cb2])
        print("fista", inner_its, run, len(fista.timing))
        print("FISTA-{}: finish run {}, time = {}".format(inner_its, run, np.sum(fista.timing)))

        dd = zarr.open_group("results/deblurring_runs_alpha_{}/fista_run_{}_warm_{}_with_pdhg_precond_optimal.zarr".format(alpha, run, inner_its))
        dd["solution"] = fista.solution.array
        dd["objective"] = fista.objective
        dd["rse"] = fista.rse
        dd["ssim"] = fista.ssim
        dd["psnr"] = fista.psnr
        dd["timing"] = fista.timing

        for prob in probs:
    
            print("ProxSkip-{}: start run {}, prob {}".format(inner_its, run, prob))
            cb2 = StoppingCriterion(epsilon=0.99e-5)
            proxskip = ProxSkip(initial = initial, f = F, step_size = step_size, g=G, 
                        update_objective_interval = 1, prob=prob, seed=40,
                        max_iteration = num_iterations) 
            proxskip.run(verbose=0, callback=[cb1, cb2]) 
            print("proxskip", prob, run, inner_its, len(proxskip.timing))
            print("ProxSkip-{}: finish run {}, prob {}, time {}".format(inner_its, run, prob, np.sum(proxskip.timing)))

            dd = zarr.open_group("results/deblurring_runs_alpha_{}/proxskip_run_{}_warm_{}_prob_{}_with_pdhg_precond_optimal.zarr".format(alpha, run, inner_its, prob))
            dd["solution"] = proxskip.solution.array
            dd["objective"] = proxskip.objective
            dd["rse"] = proxskip.rse
            dd["ssim"] = proxskip.ssim
            dd["psnr"] = proxskip.psnr
            dd["use_prox"] = proxskip.use_prox
            dd["no_use_prox"] = proxskip.no_use_prox
            dd["timing"] = proxskip.timing