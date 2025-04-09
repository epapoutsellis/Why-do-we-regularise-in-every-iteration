import cvxpy
import scipy.sparse as sp
import numpy as np



def SparseMat_GradientOperator(shape, direction='forward', order=1, boundaries='Neumann', **kwargs):

    len_shape = len(shape)    
    allMat = dict.fromkeys(range(len_shape))
    discretization = kwargs.get('discretization',[1.0]*len_shape)

    if order == 1:

        for i in range(0,len_shape):

            if direction == 'forward':

                mat = 1/discretization[i] * sp.spdiags(np.vstack([-np.ones((1,shape[i])),np.ones((1,shape[i]))]), [0,1], shape[i], shape[i], format = 'lil')

                if boundaries == 'Neumann':
                    mat[-1,:] = 0
                elif boundaries == 'Periodic':
                    mat[-1,0] = 1

            elif direction == 'backward':

                mat = 1/discretization[i] * sp.spdiags(np.vstack([-np.ones((1,shape[i])),np.ones((1,shape[i]))]), [-1,0], shape[i], shape[i], format = 'lil')

                if boundaries == 'Neumann':
                    mat[:,-1] = 0
                elif boundaries == 'Periodic':
                    mat[0,-1] = -1

            tmpGrad = mat if i == 0 else sp.eye(shape[0])

            for j in range(1, len_shape):

                tmpGrad = sp.kron(mat, tmpGrad ) if j == i else sp.kron(sp.eye(shape[j]), tmpGrad )

            allMat[i] = tmpGrad

    else:
        raise NotImplementedError    

    return allMat


def tv(u, isotropic=True, direction = "forward", boundaries = "Neumann"):
        
    G = SparseMat_GradientOperator(u.shape, direction = direction, order = 1, boundaries = boundaries)        
    DX, DY = G[1], G[0]
  
    if isotropic:
        return cvxpy.sum(cvxpy.norm(cvxpy.vstack([DX @ cvxpy.vec(u), DY @ cvxpy.vec(u)]), 2, axis = 0))
    else:
        return cvxpy.sum(cvxpy.norm(cvxpy.vstack([DX @ cvxpy.vec(u), DY @ cvxpy.vec(u)]), 1, axis = 0))



def tgv(u, w1, w2, alpha0, alpha1, boundaries = "Neumann"):

    G1 = SparseMat_GradientOperator(u.shape, direction = 'forward', order = 1, boundaries = boundaries)  
    DX, DY = G1[1], G1[0]

    G2 = SparseMat_GradientOperator(u.shape, direction = 'backward', order = 1, boundaries = boundaries) 
    divX, divY = G2[1], G2[0]
  
    return alpha0 * cvxpy.sum(cvxpy.norm(cvxpy.vstack([DX @ cvxpy.vec(u) - cvxpy.vec(w1), DY @ cvxpy.vec(u) - cvxpy.vec(w2)]), 2, axis = 0)) + \
           alpha1 * cvxpy.sum(cvxpy.norm(cvxpy.vstack([ divX @ cvxpy.vec(w1), divY @ cvxpy.vec(w2), \
                                      0.5 * ( divX @ cvxpy.vec(w2) + divY @ cvxpy.vec(w1) ), \
                                      0.5 * ( divX @ cvxpy.vec(w2) + divY @ cvxpy.vec(w1) ) ]), 2, axis = 0  ) )


def tv_denoising_primal(noisy_data, alpha, verbose=True, solver=cvxpy.MOSEK):
    
    u_cvx = cvxpy.Variable(noisy_data.shape)

    fidelity = 0.5 * cvxpy.sum_squares(u_cvx - noisy_data)   
    regulariser = alpha * tv(u_cvx)

    # objective
    obj =  cvxpy.Minimize( regulariser +  fidelity)
    prob = cvxpy.Problem(obj, constraints = [])

    # Choose solver ( SCS, MOSEK(license needed) )
    tv_cvxpy = prob.solve(verbose = verbose, solver = solver)

    return u_cvx.value, obj.value


def tv_denoising_dual(noisy_data, alpha, verbose=True, solver=cvxpy.MOSEK):
    N,M = noisy_data.shape

    q1 = cvxpy.Variable(N*M)
    q2 = cvxpy.Variable(N*M)
    t = cvxpy.Variable(N*M)

    G = SparseMat_GradientOperator(noisy_data.shape, direction = "forward", order = 1, boundaries = "Neumann")        
    DX, DY = G[1], G[0]
    
    div = -(DX.T@q1 + DY.T@q2)
    fidelity = 0.5 * cvxpy.sum_squares(div + noisy_data.flatten(order="F")) - 0.5*cvxpy.sum_squares(noisy_data)    
    constraints = [cvxpy.norm(cvxpy.vstack([q1, q2]), axis=0)<=alpha]
    
    obj_dual =  cvxpy.Minimize(fidelity)
    prob_dual = cvxpy.Problem(obj_dual, constraints)
    
    dual_tv = prob_dual.solve(verbose = verbose, solver = solver) 

    denoised_image = np.reshape(noisy_data.flatten(order="F") -  (DX.T@q1.value + DY.T@q2.value), noisy_data.shape, order="F")

    return denoised_image, q1.value, q2.value, obj_dual.value


def tv_denoising_dual_strong_convex(noisy_data, alpha, epsilon, verbose=True, solver=cvxpy.MOSEK):
    N,M = noisy_data.shape

    q1 = cvxpy.Variable(N*M)
    q2 = cvxpy.Variable(N*M)
    t = cvxpy.Variable(N*M)

    G = SparseMat_GradientOperator(noisy_data.shape, direction = "forward", order = 1, boundaries = "Neumann")        
    DX, DY = G[1], G[0]
    
    div = -(DX.T@q1 + DY.T@q2)
    fidelity = 0.5 * cvxpy.sum_squares(div + noisy_data.flatten(order="F")) - 0.5*cvxpy.sum_squares(noisy_data) +(epsilon/(2*alpha))*cvxpy.sum_squares(cvxpy.vstack([q1, q2]))
    constraints = [cvxpy.norm(cvxpy.vstack([q1, q2]), axis=0)<=alpha]
    
    obj_dual =  cvxpy.Minimize(fidelity)
    prob_dual = cvxpy.Problem(obj_dual, constraints)


    dual_tv = prob_dual.solve(verbose = verbose, solver = solver) 

    denoised_image = np.reshape(noisy_data.flatten(order="F") -  (DX.T@q1.value + DY.T@q2.value), noisy_data.shape, order="F")

    return denoised_image, q1.value, q2.value, obj_dual.value








