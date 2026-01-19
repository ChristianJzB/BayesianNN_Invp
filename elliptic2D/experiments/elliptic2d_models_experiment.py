import sys
import os
import argparse
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)  # This allows importing from base, Elliptic, etc.
sys.path.append(os.path.join(project_root, "elliptic2D"))  # Explicitly add Elliptic folder

from experiments.models_config import *
from elliptic2D.experiments.experiments_utilities import *

# Main experiment runner
def run_experiment(config_experiment,config_model,device):

    eval_save = True if config_experiment.model_type == "fem" else False

    paths = build_paths(config_model)
    
    model_name, model = get_models(config_experiment,config_model, paths, device)

    sampler ,obs_points, sol_test = get_sampler(model,config_experiment,device)

    #for i in range(config_experiment.repeat):
    if config_experiment.mcmc:
        print(f"Running MCMC for {model_name}")
        sampler.run_chain(verbose = config_experiment.verbose)           
        np.save(paths[f"{model_name}_mcmc"].format(i=0), sampler.mcmc_samples[sampler.burnin:,:].cpu().numpy())

        if eval_save:
            np.save(paths[f"{model_name}_mcmc_eval"].format(i=0), sampler.lh_mcmc_evaluations[sampler.burnin:,:].cpu().numpy())

        if not config_experiment.marginal_approximation and not config_experiment.model_type == "fem":
            np.save(paths[f"{model_name}_mean_proposal"], sampler.dt.cpu().numpy())

    if config_experiment.da:
        print(f"Running MCMC/DA for {model_name}") 
        if config_experiment.marginal_approximation:
            mean_dt = np.load(paths[f"{model_name}_mean_proposal"])
            sampler.dt = torch.tensor(mean_dt)

        sampler.run_da(verbose = config_experiment.verbose)                
        np.save(paths[f"{model_name}_da"].format(i=0), sampler.da_acc_rej.cpu().numpy())
        np.save(paths[f"{model_name}_da_chain"].format(i=0), sampler.fine_model_chain.cpu().numpy())


    if config_experiment.alpha_eval:
        print(f"Computing Error, Alpha and Time for {model_name}") 
        error_eval(model,model_name,config_experiment,config_model,obs_points,sol_test,paths,device)       

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Elliptic Experiment")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--model_type", type=str,default="FEM",required=True, help="Surrogate type")
    parser.add_argument("--kl", type=int,default=2,help="KL_expansion")
    
    parser.add_argument("--N",  type=int,default=2, help="Number of training samples")
    parser.add_argument("--kernel_spatial", type=str,default="MaternKernel52", help="Fit DeepGala")
    parser.add_argument("--kernel_parameter", type=str,default="SquaredExponential", help="Fit DeepGala")

    parser.add_argument("--dgala", action="store_true", help="Fit DeepGala")
    parser.add_argument("--hidden_layers", type=int,default=2,help="Number of layers")
    parser.add_argument("--num_neurons", type=int,default=20,help="Number of neurons/layer")
    parser.add_argument("--batch_size", type=int,default=10,help="Mini batche size")

    parser.add_argument("--proposal", type=str,default="random_walk",help="MCMC Proposal")
    parser.add_argument("--marginal_approximation", action="store_true", help="Run DA-MCMC for NN")
    parser.add_argument("--mcmc", action="store_true", help="Run MCMC")
    parser.add_argument("--da", action="store_true", help="Run DA-MCMC")
    parser.add_argument("--mcmc_samples", type=int,default=2_500_000,help="#Samples for the MCMC")
    parser.add_argument("--da_eval", type=int,default=100_000,help="#Evaluations for the DA-MCMC")
    parser.add_argument("--alpha_eval", action="store_true",help="Computes Alpha, Error and Time")

    parser.add_argument("--repeat", type=int,default=3, help="Repeat MCMC ntimes")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.model_type == "pigp":
        config_model = elliptic2d_pigp_experiment(kernel_spatial = args.kernel_spatial,kernel_parameter=args.kernel_parameter,
                                                N=args.N,kl=args.kl,marginal = args.marginal_approximation)
    elif args.model_type == "gp":
        config_model = elliptic2d_gp_experiment(kernel=args.kernel_parameter,N= args.N,kl=args.kl,marginal = args.marginal_approximation)
    elif args.model_type == "dgala":
        config_model = elliptic2d_neural_training(args.N,args.hidden_layers,args.num_neurons,args.batch_size,args.kl,
                                                args.dgala,marginal = args.marginal_approximation)
    elif args.model_type == "fem":
        print("Computing MCMC samples from the reference model")
        config_model = elliptic_fem_experiment(kl=args.kl)
    else:
        raise ValueError(f"Model '{args.model_type}' is not supported.")

    config_experiment = elliptic2d_inverse_problem(args.model_type,args.verbose,args.dgala,args.mcmc,args.da,
                            args.marginal_approximation,args.alpha_eval,args.kl,args.proposal,args.mcmc_samples,args.da_eval, args.repeat)
    
    config_model.seed = config_experiment.seed
    
    run_experiment(config_experiment,config_model,device)
