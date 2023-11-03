from abc import ABC, abstractmethod
from opacus.grad_sample import GradSampleModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
from scipy.stats import ncx2, norm
import gmip
import math
## Implementation of the Gradient Likelihood Ratio Attack (GLiR)

class GradientInterface(ABC):

    @abstractmethod
    def compute_gradients(self, *input_tensors):
        """ Compute the gradients for a batch of input tensors. Usually these are expected
            to have the same length (e.g., input data at index 0, labels as index 1).
        """
        pass

    @abstractmethod
    def get_model(self):
        """ return the underlying classification model, such that it can be updated,
            e.g., after performing training steps.
            the object returned by this function is passed to model update interface
        """
        pass


class ModelTracingInterface(ABC):
    """ An interface that allows to set models back to a certain point in the training process,
        for instance using stored weights for each step, or does the actual training steps on the unkonwn data set
        and update the corresponding model
    """

    @abstractmethod
    def update_model_to_next_step(self, model):
        """ 
            Update the model parameters to reflect the next step of training.

        """
        pass

    @abstractmethod
    def get_gradients_for_step(self):
        """ Get the recorded gradient for the current step in training. """
        pass


class ClassificationModelGradients(GradientInterface):
    """ 
        An implementation of the Gradient Interface.
    """
    def __init__(self, model: GradSampleModule, criterion, cutoff_threshold=float("inf"), device="cuda"):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.cutoff = cutoff_threshold

    def compute_gradients(self, *input_tensors):
        """ Compute model gradients for the model """
        self.__zero_grad()
        inputs, labels = input_tensors[0].to(self.device), input_tensors[1].to(self.device)
        #print(type(self.model), self.model.fc.weight.requires_grad)
        outputs_batch = self.model(inputs)
        #print(outputs_batch.shape)
        loss_batch = self.criterion(outputs_batch, labels) # Note that loss should return one element per batch item.
        loss_batch.backward()
        grads = self.__aggregate_grads()
        #print(grads.shape)
        grad_norms = grads.norm(dim=1)
        #print(grad_norms, cutoff)
        grads[grad_norms>self.cutoff] = grads[grad_norms>self.cutoff]/grad_norms[grad_norms>self.cutoff].reshape(-1,1)
        #self.model.remove_hooks()
        return grads

    def __zero_grad(self):
        for t in self.model.parameters():
            t.grad_sample = None
            t.grad_summed = None
            t.grad = None

    def __aggregate_grads(self):
        param_grad_list = []
        for t in self.model.parameters():
            if t.requires_grad == False:
                continue
            #print(type(t.grad_sample), type(t))
            if t.grad_sample is not None:
                param_grad_list.append(t.grad_sample.reshape(len(t.grad_sample), -1).clone().cpu())
        return torch.cat(param_grad_list, axis=1)

    def get_model(self):
        return self.model

class DirectGradients(GradientInterface):
    """ Implementation of the interface that directly returns tha gradients that were passed.
        Use in the simulation study.
    """
    def compute_gradients(self, *input_tensors):
        """ Compute the gradients for a batch of input tensors. Usually these are expected
            to have the same length (e.g., input data at index 0, labels as index 1).
        """
        return input_tensors[0]

    def get_model(self):
        """ return the underlying classification model, such that it can be updated,
            e.g., after performing training steps.
            the object returned by this function is passed to model update interface
        """
        return None

class CheckpointListTracer(ModelTracingInterface):
    """ 
        Trace the model using logfiles created by the training scripts in this project.
    """
    def __init__(self, path_to_trace_file, custom_load_function=None):
        """
            path_to_trace_file: The path to the log fiel
            custom_load_function: a function(model, state_dict) that takes a dict and updates a model accordingly
            similar to torch.nn.Module.load_state_dict, but may be implemented differently if only parts of the model are
            trainable parameters.
        """
        self.res_dict = None
        self.path_to_trace_file = path_to_trace_file
        self.res_dict = torch.load(path_to_trace_file)
        self.custom_load_function = custom_load_function
        self.update_cnt = 0

    def update_model_to_next_step(self, model):
        """ Compute the gradients for a batch of input tensors. Usually these are expected
            to have the same length (e.g., input data at index 0, labels as index 1).
        """
        self.custom_load_function(model, self.res_dict["stepwise_params"][self.update_cnt])
        self.update_cnt += 1

    def get_gradients_for_step(self):
        """ Get the gradient for a certain step in training. """
        return torch.cat([t.flatten().cpu() for t in self.res_dict["stepwise_grads"][self.update_cnt-1]])

    def get_used_sample_idx(self):
        """ return the index list of the samples used in training from the training dataset. The 
            remaining samples can be used as test points or background data
        """
        return self.res_dict["samples_used"]

class SimulatedGradientTracer(ModelTracingInterface):
    """ A simulated multi-variate gaussian distribution of gradients.
        Note that the the distribution does not change, as we only use this interface to simulate the one-step attack.
    """
    def __init__(self, mygrad):
        self.mygrad = mygrad
    
    def update_model_to_next_step(self, model):
        """ Compute the gradients for a batch of input tensors. Usually these are expected
            to have the same length (e.g., input data at index 0, labels as index 1).
        """
        pass

    def get_gradients_for_step(self):
        """ Get the gradient for a certain step in training. """
        return self.mygrad

class GaussianDataLoader(DataLoader):
    class GaussianLoaderIter():
        def __init__(self, parent):
            self.parent = parent

        def __next__(self,):
            return (self.parent.s_half @ torch.randn(len(self.parent.mu), self.parent.batch_size)).t()

    def __init__(self, mu, Sigma, batch_size, mylen=int(1e9)):
        self.batch_size = batch_size
        self.mu = mu
        self.eigvals, self.eigvects = torch.linalg.eigh(Sigma)
        s_half = (self.eigvects @ torch.diag(torch.sqrt(self.eigvals)) @ self.eigvects.t())
        self.s_half = s_half # Sigma^{1/2}
        self.mylen = mylen

    def __iter__(self):
        return GaussianDataLoader.GaussianLoaderIter(self)

    def __len__(self):
        return self.mylen

class GLiRAttack():
    """ Run the gradient likelihood ratio attack (GLiR). """

    def __init__(self, background_loader: DataLoader, gradient_function: GradientInterface, 
            model_tracer: ModelTracingInterface, num_params: int, training_batch_size: int, n_background_samples = 10000):
        """
            initialize the object.
            background_loader: torch.data_loader for the background dataset (the training data distribution)
        """
        self.background_loader = background_loader
        self.gradient_function = gradient_function
        self.model_tracer = model_tracer
        self.d = num_params
        self.n = training_batch_size
        self.n_background_samples = n_background_samples

    def compute_glir_attack_scores_w_loader(self, dataset_loader, n_load, n_steps=1):
        """ 
            Run the GLiR attack on a number of points and compute scores.
        """
        tot_scores_mat = []
        for step in range(n_steps):
            self.model_tracer.update_model_to_next_step(self.gradient_function.get_model())
            m = self.model_tracer.get_gradients_for_step()
            score_in = self._compute_step_attack_scores(m, dataset_loader, n_load)
            tot_scores_mat.append(score_in.reshape(1, -1))

        return torch.cat(tot_scores_mat, axis=0).sum(axis=0)

    def compute_glir_attack_scores(self, *input_tensors, batch_size=16, n_steps=1):
        """ compute attack scores for tensors of points as inputs.
            batch_size: batch size to use for input point gradient computation
        """
        mydataset = TensorDataset(*input_tensors)
        n_load = len(mydataset)
        data_loader = DataLoader(mydataset, batch_size=batch_size)
        return self.compute_glir_attack_scores_w_loader(data_loader, n_load, n_steps=n_steps)

    def _compute_cdf_scores(self, k_in, mean_in):
        """ Compute the p-values under the nullhypotheses: x' is a test point. 
            Return log of p-value.
        """
        #print("got n_avg=", self.n, "d=", self.d)
        gamma_2 = self.n*k_in
        qlist = []
        for i in range(len(gamma_2)):
            q = ncx2.logcdf(mean_in[i], self.d, gamma_2[i])
            #print(q)
            qlist.append(max(q, -200.0))
        score_in = torch.tensor(qlist)
        
        return score_in

    def _estimate_grads(self, data_loader_use, n_estimation_samples=100):
        n_samples = 0
        loader_iter = iter(data_loader_use)
        grad_list = []
        while n_samples < n_estimation_samples:
            data = next(loader_iter)
            grad_list.append(self.gradient_function.compute_gradients(*data))
            n_samples += len(data[0])
        all_grad = torch.cat(grad_list, axis=0)[:n_estimation_samples]
        return all_grad

    def _compute_step_attack_scores(self, recorded_step_mean, dataloader, n_load):
        """ 
            Compute the log likelihoods for the density attack.
            model: 
        """
        # model_opacus = GradSampleModule(model, loss_reduction = "mean").to(use_device) ##?
        #print(str(type(self.background_loader)))
        #print("Computing point grads ...")
        grads_in = self._estimate_grads(dataloader, n_estimation_samples=n_load) # Obtain gradients of query points

        # Computation of K and S (test statistic)
        if str(type(self.background_loader)) == "<class 'gmip.glir.GaussianDataLoader'>": 
            ## Simulated gradients, get true distribution params
            ## Invert using eigenvalues
            grad_means = self.background_loader.mu.reshape(1, -1)
            evects, eigvals = self.background_loader.eigvects, self.background_loader.eigvals
            # compute K via Sigma^{-1/2} which is numerically more stable
            sigma_inv_half = (evects @ torch.diag(1.0/torch.sqrt(eigvals)) @ evects.t())
            k_in = torch.sum((sigma_inv_half @ grads_in.t()).pow(2), axis=0)
            diff = (recorded_step_mean.reshape(-1,1) - grads_in.t())
            diff_trans = math.sqrt(self.n-1)*(sigma_inv_half @ diff)
            mean_in = torch.sum(diff_trans.pow(2), axis=0)
        else:
            ## Real gradients, estimate distribution parameters
            grad_estimation = self._estimate_grads(self.background_loader, n_estimation_samples=self.n_background_samples)
            grad_means = grad_estimation.mean(axis=0, keepdim=True)
            grad_vars_norm = grad_estimation - grad_means
            sigma_diag = grad_vars_norm.var(axis=0)
            sigma = (grad_vars_norm.t() @ grad_vars_norm)/len(grad_vars_norm)
            print("Inverting Sigma...")
            kinv = torch.cholesky_inverse(sigma)
            # calculate_test_stats
            mean_in = (self.n-1)*torch.sum((recorded_step_mean.reshape(1,-1)-grads_in).t() * (kinv @ (recorded_step_mean.reshape(1,-1)-grads_in).t()), axis=0)
            k_in = torch.sum((grads_in - grad_means).t() * (kinv @ (grads_in - grad_means).t()), axis=0)
       


        return self._compute_cdf_scores(k_in, mean_in)