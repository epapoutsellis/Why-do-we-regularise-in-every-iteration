## import libraries
from cil.framework import AcquisitionGeometry, ImageGeometry
from cil.plugins.astra import FBP, ProjectionOperator
from cil.optimisation.algorithms import PDHG, FISTA
from cil.optimisation.functions import L2NormSquared, LeastSquares, BlockFunction, MixedL21Norm, IndicatorBox
from cil.optimisation.operators import BlockOperator, GradientOperator
from cil.processors import TransmissionAbsorptionConverter, Slicer, Binner
import zarr
import matplotlib.pyplot as plt
from cil.io import ZEISSDataReader
import numpy as np
from MixedL21Norm_fix import MixedL21Norm_fix

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
A = ProjectionOperator(ig2D, ag2D, device="cpu")
f1 = 0.5 * L2NormSquared(b=data2D)
f2 = alpha * MixedL21Norm_fix()
F = BlockFunction(f1, f2)
Grad = GradientOperator(ig2D)
K = BlockOperator(A, Grad)
G = IndicatorBox(lower=0)
normK = K.norm()
initial = ig2D.allocate()

tmp_sigma = K.range_geometry().allocate()
tmp_sigma.get_item(0).fill(A.direct(A.domain_geometry().allocate(1)))
tmp_sigma.get_item(1).get_item(0).fill(Grad.domain_geometry().allocate(2))
tmp_sigma.get_item(1).get_item(1).fill(Grad.domain_geometry().allocate(2))
sigma = 400./tmp_sigma

tmp_tau = K.domain_geometry().allocate()
tmp_tau.fill(A.adjoint(A.range_geometry().allocate(1))
             + Grad.domain_geometry().allocate(2) + Grad.domain_geometry().allocate(2))
tau = 0.0025/tmp_tau

pdhg_optimal = PDHG(initial = initial, f=F, g=G, operator=K, sigma=sigma, tau=tau,
            max_iteration = 200000,
            update_objective_interval=1000)    
pdhg_optimal.run(verbose=1)

dd = zarr.open_group("results/pdhg_optimal_finden_tv_alpha_{}_explicit_precond_maxiterations_{}.zarr".format(alpha, 200000))
dd["solution"] = pdhg_optimal.solution.array
dd["objective"] = pdhg_optimal.objective
dd["pdgap"] = pdhg_optimal.primal_dual_gap