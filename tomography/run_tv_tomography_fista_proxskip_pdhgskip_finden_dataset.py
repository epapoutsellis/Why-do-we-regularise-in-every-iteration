## import libraries
from cil.framework import AcquisitionGeometry, ImageGeometry
from cil.plugins.astra import FBP, ProjectionOperator
from cil.optimisation.algorithms import PDHG, FISTA, PD3O
from cil.optimisation.functions import L2NormSquared, LeastSquares, MixedL21Norm, IndicatorBox, ZeroFunction, SGFunction, SAGAFunction, BlockFunction
from cil.optimisation.operators import BlockOperator, GradientOperator
from cil.optimisation.utilities import MetricsDiagnostics, StatisticsDiagnostics, AlgorithmDiagnostics, RSE
from cil.processors import TransmissionAbsorptionConverter, Slicer, Binner
import zarr
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markersize'] = 12
plt.rcParams['font.size'] = 20
from cil.io import ZEISSDataReader
from cil.plugins.astra.utilities import convert_geometry_to_astra, convert_geometry_to_astra_vec_2D
from TotalVariation import TotalVariationNew
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import numpy as np
from cil.optimisation.algorithms import PDHGSkip, ProxSkip


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


def save_to_zarr(algorithm_instance, name, path):
    
    dd = zarr.open_group(path)
    dd["solution"] = algorithm_instance.solution.array
    dd["rse"] = algorithm_instance.rse
    dd["ssim"] = algorithm_instance.ssim
    dd["psnr"] = algorithm_instance.psnr  
    dd["timing"] = algorithm_instance.timing   

    if name=="FISTA":
        dd["objective"] = [algorithm_instance.f(i) + algorithm_instance.g(i) for i in algorithm_instance.list_iterates]
    elif name=="ProxSkip":
        dd["objective"] = [algorithm_instance.f(i) + algorithm_instance.g(i) for i in algorithm_instance.list_iterates]
        dd["use_prox"] = algorithm_instance.use_prox
        dd["no_use_prox"] = algorithm_instance.no_use_prox        
    elif name=="PDHGSkip":
        dd["objective"] = [algorithm_instance.h(algorithm_instance.operator.direct(i)) + algorithm_instance.g(i) for i in algorithm_instance.list_iterates]
        dd["use_prox"] = algorithm_instance.use_prox
        dd["no_use_prox"] = algorithm_instance.no_use_prox      

def update_objective(self):
    self.list_iterates.append(self.get_output().copy())

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def NRSE(x, y, **kwargs):
    """
     root squared error between two numpy arrays
    """
    return np.sqrt(np.sum(np.abs(x - y*mask.ravel())**2))/np.sqrt(np.sum(x**2))

FISTA.update_objective = update_objective
ProxSkip.update_objective = update_objective

sino = zarr.load("../data/tomography/NiPd_spent_01_microct_rings_removed_2D.zarr")

print(f"sinogram shape is {sino.shape}")

num_angles, horizontal = sino.shape

angles_list = np.linspace(0, np.pi, num_angles)
ag2D = AcquisitionGeometry.create_Parallel2D().\
        set_panel(horizontal).\
        set_angles(angles_list, angle_unit="radian").\
        set_labels(['angle','horizontal'])
ig2D = ag2D.get_ImageGeometry()

data2D = ag2D.allocate()
data2D.fill(sino)


print(ig2D)
print(ag2D)

alpha = 0.1

pdhg_optimal_info = zarr.load("results/pdhg_optimal_finden_tv_alpha_{}_explicit_precond_maxiterations_200000.zarr".format(alpha))
pdhg_optimal_np = pdhg_optimal_info["solution"]
pdhg_optimal_cil = ig2D.allocate()
pdhg_optimal_cil.fill(pdhg_optimal_np)


h, w = pdhg_optimal_cil.shape
mask = create_circular_mask(h, w, center=(170,165), radius=150)

SSIM_dr = lambda x, y: SSIM(x, y*mask.ravel(), data_range=x.max()-x.min())
PSNR_dr = lambda x, y: PSNR(x, y*mask.ravel(), data_range=x.max()-x.min())



cb1 = MetricsDiagnostics(reference_image=pdhg_optimal_cil*mask, 
                         metrics_dict={"rse":NRSE, "ssim":SSIM_dr, "psnr":PSNR_dr})  

A = ProjectionOperator(ig2D, ag2D, device="cpu")
Anorm = A.norm()


Gnorm10 = alpha * TotalVariationNew(max_iteration=10, tolerance=None, correlation='Space',
    backend='c', lower=0, upper=np.infty, isotropic=True, split=False, info=False, strong_convexity_constant=0,
    warm_start=True)

Gnorm50 = alpha * TotalVariationNew(max_iteration=50, tolerance=None, correlation='Space',
    backend='c', lower=0, upper=np.infty, isotropic=True, split=False, info=False, strong_convexity_constant=0,
    warm_start=True)

Gnorm100 = alpha * TotalVariationNew(max_iteration=100, tolerance=None, correlation='Space',
    backend='c', lower=0, upper=np.infty, isotropic=True, split=False, info=False, strong_convexity_constant=0,
    warm_start=True)

G_dict = {100:Gnorm100, 50:Gnorm50, 10:Gnorm10}

F_fista = LeastSquares(A=A, b=data2D, c=0.5)
F_pdhg = 0.5*L2NormSquared(b = data2D)
step_size = 1./F_fista.L
initial = ig2D.allocate()

max_iteration = 4

runs = 10
probs = [1.0, 0.7, 0.5, 0.3, 0.1]


for run in range(runs):

    for inner_its, G in G_dict.items():
    
        print("Run FISTA with step size={}, run={}, warm={}".format(step_size, run, inner_its))
        cb2 = StoppingCriterion(epsilon=0.99e-5)
        fista = FISTA(initial = initial, f = F_fista, step_size = step_size, g=G, 
                    update_objective_interval = 1, 
                    max_iteration = max_iteration) 
        fista.list_iterates = []
        fista.run(verbose=0, callback=[cb1, cb2])
        
        save_to_zarr(fista, "FISTA", "results/tomography_from_pdhg_explicit_precon_finden_runs_alpha_{}/fista_step_size_{}_run_{}_warm_{}.zarr".format(alpha, step_size, run, inner_its))
        print("Finish")    
    
        for prob in probs:
    
            print("Run Proxskip with step size=2.0, prob={}, run={}, warm={}".format(prob, run, inner_its))
            cb2 = StoppingCriterion(epsilon=0.99e-5)
            proxskip = ProxSkip(initial = initial, f = F_fista, step_size = 1.99*step_size, g=G, 
                        update_objective_interval = 1, prob=prob, seed=40,
                        max_iteration = max_iteration) 
            proxskip.list_iterates = []
            proxskip.run(verbose=0, callback=[cb1, cb2])             
            save_to_zarr(proxskip, "ProxSkip", "results/tomography_from_pdhg_explicit_precon_finden_runs_alpha_{}/proxskip_prob_{}_step_size_2.0x_run_{}_warm_{}.zarr".format(alpha, prob, run, inner_its))
            print("Finish")
                    
            for tau in [0.08]:
                gamma = 1./(tau*Anorm**2)
                print("Run PDHGSkip with tau={}, prob={}, run={}, warm={}".format(tau, prob, run, inner_its))
                cb2 = StoppingCriterion(epsilon=0.99e-5)
                pdhgskip = PDHGSkip(initial=initial, g=G, h=F_pdhg, operator=A, 
                                                      prob=prob, max_iteration=max_iteration,
                                                      update_objective_interval=1, gamma=gamma, tau=tau, seed=40)
                pdhgskip.list_iterates = []
                pdhgskip.run(verbose=0, callback=[cb1, cb2])
                print(np.sum(pdhgskip.timing), gamma, prob, pdhgskip.rse[-1])
                save_to_zarr(pdhgskip, "PDHGSkip", "results/tomography_from_pdhg_explicit_precon_finden_runs_alpha_{}/pdhgskip_prob_{}_step_size_{}_run_{}_warm_{}.zarr".format(alpha, prob, tau, run, inner_its))
                print("Finish")    