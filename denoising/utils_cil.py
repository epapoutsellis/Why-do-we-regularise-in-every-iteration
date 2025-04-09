from cil.optimisation.operators import LinearOperator
from cil.optimisation.functions import Function
from cil.optimisation.utilities import MetricsDiagnostics, StatisticsDiagnostics, AlgorithmDiagnostics, RSE
import inspect
import numpy as np

class AdjointOperator(LinearOperator):
    
    def __init__(self, operator):
        super(AdjointOperator, self).__init__(domain_geometry=operator.range_geometry(), 
                                       range_geometry=operator.domain_geometry())         
        self.operator = operator
                
    def direct(self, x, out=None):
        return self.operator.adjoint(x, out=out)
        
    def adjoint(self, x, out=None):
        return self.operator.direct(x, out=out)
    
class IndicatorMixedL21Norm(Function):
    
    def __init__(self, function):
        
        self.function = function
        super(IndicatorMixedL21Norm).__init__()
        
    def __call__(self,x):
        
        # should check for the norm set inf or 0. Return 0 atm.
        return 0.0

    def convex_conjugate(self,x):
        
        # should check for the norm set inf or 0. Return 0 atm.
        return 0.
    
    def proximal(self, x, tau, out=None):
        
        return self.function.proximal_conjugate(x, tau, out=out)      
          

class DistanceFromPrimalSolution(MetricsDiagnostics):

    def __init__(self, reference_solution, data, operator, metrics_dict, verbose=1):

        # reference image as numpy (level) array
        self.reference_solution = reference_solution
        self.data = data
        self.operator = operator
        self.metrics_dict = metrics_dict
        self.computed_metrics = []    
        self.data_range = data.max() - data.min()

        super(MetricsDiagnostics, self).__init__(verbose=verbose)     
    
    def __call__(self, algo):
            
        test_image_array = self.data + self.operator.direct(algo.get_output())
            
        for metric_name, metric_func in self.metrics_dict.items():

            if not hasattr(algo, metric_name):
                setattr(algo, metric_name, [])   
                
            metric_list = getattr(algo, metric_name)

            metric_signature = inspect.signature(metric_func)

            if 'data_range' in metric_signature.parameters:
                metric_value = metric_func(self.reference_solution.as_array().ravel(), test_image_array.as_array().ravel(), data_range = self.data_range)                
            else:
                metric_value = metric_func(self.reference_solution.as_array().ravel(), test_image_array.as_array().ravel())

            metric_list.append(metric_value)
            
            self.computed_metrics.append(metric_value)


class TimeStoppingCriterion(AlgorithmDiagnostics):

    def __init__(self, total_time=60.):

        self.total_time = total_time
        super(TimeStoppingCriterion, self).__init__(verbose=0)

        self.should_stop = False

    def _should_stop(self):

        return self.should_stop       
            
    def __call__(self, algo):

        if algo.iteration==1:
            algo.should_stop = self._should_stop  

        ddd = np.sum(algo.timing)
        if ddd>self.total_time:
            self.should_stop = True  
            print("Stop at {} time {} ".format(algo.iteration, np.sum(algo.timing)))


