from cil.optimisation.functions import Function
from cil.framework import BlockDataContainer
import numpy as np


class MixedL21Norm_fix(Function):
    
    
    """ MixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, **kwargs):

        super(MixedL21Norm_fix, self).__init__()  
                    
        
    def __call__(self, x):
        
        r"""Returns the value of the MixedL21Norm function at x. 

        :param x: :code:`BlockDataContainer`                                           
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
              
        return x.pnorm(p=2).sum()                                
            
                            
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the MixedL21Norm function at x.
        
        This is the Indicator function of :math:`\mathbb{I}_{\{\|\cdot\|_{2,\infty}\leq1\}}(x^{*})`,
        
        i.e., 
        
        .. math:: \mathbb{I}_{\{\|\cdot\|_{2, \infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x\|_{2, \infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}
        
        where, 
        
        .. math:: \|x\|_{2,\infty} = \max\{ \|x\|_{2} \} = \max\{ \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}\}
        
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
                                        
        tmp = (x.pnorm(2).max() - 1)
        if tmp<=1e-5:
            return 0
        else:
            return np.inf
                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """

        # Note: we divide x/tau so the cases of both scalar and 
        # datacontainers of tau to be able to run
        
        
        if out is None:
            tmp = (x/tau).pnorm(2)
            res = (tmp - 1).maximum(0.0) * x/tmp

            # TODO avoid using numpy, add operation in the framework
            # This will be useful when we add cupy 
                                 
            for el in res.containers:

                elarray = el.as_array()
                elarray[np.isnan(elarray)]=0
                el.fill(elarray)

            return res
            
        else:
            
            try:
                x.divide(tau,out=x)
                tmp = x.pnorm(2)
                x.multiply(tau,out=x)
            except TypeError:
                x_scaled = x.divide(tau)
                tmp = x_scaled.pnorm(2)
 
            tmp_ig = 0.0 * tmp
            (tmp - 1).maximum(0.0, out = tmp_ig)
            tmp_ig.multiply(x, out = out)
            out.divide(tmp, out = out)
            
            for el in out.containers:
                
                elarray = el.as_array()
                elarray[np.isnan(elarray)]=0
                el.fill(elarray)  

            out.fill(out)
