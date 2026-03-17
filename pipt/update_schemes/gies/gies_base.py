"""
EnRML type schemes
"""
# External imports
import pipt.misc_tools.analysis_tools as at
import pipt.misc_tools.extract_tools as extract
import pipt.misc_tools.ensemble_tools as entools
import pipt.misc_tools.data_tools as dtools

from geostat.decomp import Cholesky
from pipt.loop.ensemble import Ensemble
from pipt.update_schemes.update_methods_ns.subspace_update import subspace_update
from pipt.update_schemes.update_methods_ns.full_update import full_update
from pipt.update_schemes.update_methods_ns.approx_update import approx_update
import sys
import pkgutil
import inspect
import numpy as np
import copy as cp
from scipy.linalg import cholesky, solve, inv, lu_solve, lu_factor

import importlib.util

# List all available packages in the namespace package
# Import those that are present
import pipt.update_schemes.gies as ns_pkg
tot_ns_pkg = []
# extract all class methods from namespace
for finder, name, ispkg in pkgutil.walk_packages(ns_pkg.__path__):
    if 'update' in name:
        spec = finder.find_spec(name)
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        tot_ns_pkg.extend(inspect.getmembers(_module, inspect.isclass))

# Internal imports
from pipt.misc_tools.analysis_tools import aug_state


class GIESMixIn(Ensemble):
    """
    This is a base template for implementating the generalized iterative ensemble smoother (GIES) in the following papers:
    Luo, Xiaodong. "Novel iterative ensemble smoothers derived from a class of generalized cost functions."
                    Computational Geosciences 25.3 (2021): 1159-1189.
    Luo, Xiaodong, and William C. Cruz. "Data assimilation with soft constraints (DASC) through a generalized iterative
                    ensemble smoother." Computational Geosciences 26.3 (2022): 571-594.
    """

    def __init__(self, keys_da, keys_en, sim):
        """
        The class is initialized by passing the PIPT init. file upwards in the hierarchy to be read and parsed in
        `pipt.input_output.pipt_init.ReadInitFile`.
        """
        # Pass the init_file upwards in the hierarchy
        super().__init__(keys_da, keys_en, sim)

        if self.restart is False:
            # Save prior state in separate variable
            #self.prior_state = cp.deepcopy(self.state)
            self.prior_enX = cp.deepcopy(self.enX) # not sure if this is wise!

            # Set parameters needed for LM-EnRML
            options = self.keys_da['iteration']
            if isinstance(options, list):
                options = extract.list_to_dict(options)

            self.data_misfit_tol = options.get('data_misfit_tol', 0.01) # default value corresping to 1%
            self.trunc_energy = options.get('trunc_energy', 0.95)
            self.step_tol = options.get('step_tol', 0.01)
            self.lam = options.get('lambda', 1.0)
            self.gamma = options.get('lambda_factor', 2.0)
            self.iteration = 0

            # Ensure that it is given as percentage
            if self.trunc_energy > 1:
                self.trunc_energy /= 100.

            # Initalize some variables
            self.prev_data_misfit = None  # Data misfit at previous iteration

            # Load ACTNUM if given
            if 'actnum' in self.keys_da.keys():
                try:
                    self.actnum = np.load(self.keys_da['actnum'])['actnum']
                except:
                    print('ACTNUM file cannot be loaded!')
            else:
                self.actnum = None

            # At the moment, the iterative loop is treated as an iterative smoother and thus we check if assim. indices
            # are given as in the Simultaneous loop.
            self.check_assimindex_simultaneous()
            self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]

            # define the list of datatypes
            self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(self.obs_data, self.assim_index)

            # Get the perturbed observations and observation scaling
            self.data_random_state = cp.deepcopy(np.random.get_state())
            self.vecObs, self.enObs = self.set_observations()

            # Get state scaling and svd of scaled prior
            self._ext_scaling()

    def calc_analysis(self):
        """
        Calculate the update step in LM-EnRML, which is just the Levenberg-Marquardt update algorithm with
        the sensitivity matrix approximated by the ensemble.
        """
        # Get Ensemble of predicted data
        _, self.enPred = at.aug_obs_pred_data(
            self.obs_data,
            self.pred_data,
            self.assim_index,
            self.list_datatypes
        )

        if self.iteration == 1:  # first iteration

            # Calculate the prior data misfit
            data_misfit = at.calc_objectivefun(
                pert_obs=self.enObs,
                pred_data=self.enPred,
                Cd=self.cov_data
            )

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = np.mean(data_misfit)
            self.prior_data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            # Log initial data misfit
            self.log_update(success=True, prior_run=True)

        if 'localanalysis' in self.keys_da:
            self.local_analysis_update()
        else:
            # Check for adjoint
            if hasattr(self, 'adjoints'):
                enAdj = dtools.combine_ensemble_dataframes(self.adjoints)
                enAdj = dtools.dataframe_to_matrix(enAdj)  # Shape (nd, ne, nx)
            else:
                enAdj = None

            # Perform the update
            self.update(
                enX=self.enX,
                enY=self.enPred,
                enE=self.enObs,
                # kwargs
                prior=self.prior_enX,
                enAdj=enAdj
            )

            # Update the state ensemble and weights
            if hasattr(self, 'step'):
                self.enX_temp = self.enX[:, 0:self.ne] + self.step
            if hasattr(self, 'w_step'):
                self.W = self.current_W + self.w_step
                self.enX_temp = np.dot(self.prior_enX, (np.eye(self.ne) + self.W / np.sqrt(self.ne - 1)))

            # Ensure limits are respected
            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.idX.keys()}
            self.enX_temp = entools.clip_matrix(self.enX_temp, limits, self.idX)

    def check_convergence(self):
        """
        Check if GIES has converged based on evaluation of change sizes of objective function, state and damping
        parameter. 

        Returns
        -------
        conv: bool
            Logic variable telling if algorithm has converged
        why_stop: dict
            Dict. with keys corresponding to conv. criteria, with logical variable telling which of them that has been
            met
        """

        # Get Ensemble of predicted data
        _, enPred = at.aug_obs_pred_data(
            self.obs_data,
            self.pred_data,
            self.assim_index,
            self.list_datatypes
        )

        # Initialize the initial success value
        success = False

        # if inital conv. check, there are no prev_data_misfit
        self.prev_data_misfit = self.data_misfit
        self.prev_data_misfit_std = self.data_misfit_std

        # Calc. std dev of data misfit (used to update lamda)
        # mat_obs = np.dot(obs_data_vector.reshape((len(obs_data_vector),1)), np.ones((1, self.ne))) # use the perturbed
        # data instead.

        #data_misfit = at.calc_objectivefun(self.enObs, enPred, self.cov_data)
        data_misfit = at.calc_objectivefun(enPred[:, :self.ne], self.vecObs[:, None], self.cov_data)

        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # # Calc. mean data misfit for convergence check, using the updated state variable
        # self.data_misfit = np.dot((mean_preddata - obs_data_vector).T,
        #                      solve(cov_data, (mean_preddata - obs_data_vector)))

        # Convergence check: Relative step size of data misfit or state change less than tolerance
        if abs(1 - (self.data_misfit / self.prev_data_misfit)) < self.data_misfit_tol:
                #or self.lam >= self.lam_max:
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit / self.prev_data_misfit) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit,
                        'prev_data_misfit': self.prev_data_misfit,
                        'lambda': self.lam}
            if hasattr(self, 'lam_max'):
                why_stop['lambda_stop'] = (self.lam >= self.lam_max)

            if self.data_misfit >= self.prev_data_misfit:
                success = False
                self.logger.info(
                    f'Iterations have converged after {self.iteration} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit:0.2f} to {self.prev_data_misfit:0.2f}')
            else:
                self.logger.info(
                    f'Iterations have converged after {self.iteration} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit:0.2f} to {self.data_misfit:0.2f}')
            # Return conv = True, why_stop var.
            return True, success, why_stop

        else:  # conv. not met
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit / self.prev_data_misfit) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit,
                        'prev_data_misfit': self.prev_data_misfit,
                        'lambda': self.lam}
            if hasattr(self, 'lam_max'):
                why_stop['lambda_stop'] = (self.lam >= self.lam_max)

            ###############################################
            ##### update Lambda step-size values ##########
            ###############################################
            # If reduction in mean data misfit, reduce damping param
            if self.data_misfit < self.prev_data_misfit:
                # Reduce damping parameter (divide calculations for ANALYSISDEBUG purpose)
                if not hasattr(self, 'lam_min'):
                    self.lam = self.lam / self.gamma
                else:
                    if self.lam > self.lam_min:
                        self.lam = self.lam / self.gamma

                success = True

                #self.current_state = cp.deepcopy(self.state)

                # Update state ensemble
                self.enX = cp.deepcopy(self.enX_temp)
                self.enX_temp = None

                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)

            else:  # Reject iteration, and increase lam
                # Increase damping parameter (divide calculations for ANALYSISDEBUG purpose)
                self.lam = self.lam * self.gamma
                success = False

            self.logger.info(f'Iter {self.iteration}: <Obj. func. val. (mean +/- STD): '
                             f'{self.data_misfit:.2f} +/- {self.data_misfit_std:.2f}'
                             ' | '
                             f'Mean value reduced by {100 * (1 - (self.data_misfit / self.prev_data_misfit)):.2f}%'
                             ' | '
                             f'STD value reduced by {100 * (1 - (self.data_misfit_std / self.prev_data_misfit_std)):.2f}%'
                             '|'
                             'Lamba for next iteration:' f'{self.lam:.2e}>')

            # Log update results
            self.log_update(success=success)

            if not success:
                # Reset the objective function after report
                self.data_misfit = self.prev_data_misfit
                self.data_misfit_std = self.prev_data_misfit_std

            # Return conv = False, why_stop var.
            return False, success, why_stop

    def log_update(self, success, prior_run=False):
        '''
        Log the update results in a formatted table.
        '''
        log_data = {
            "Iteration": f'{0 if prior_run else self.iteration}',
            "Status": "Success" if (prior_run or success) else "Failed",
            "Data Misfit": self.data_misfit,
            "λ": self.lam
        }
        if not prior_run:
            if success:
                log_data["Reduction (%)"] = 100 * (1 - self.data_misfit / self.prev_data_misfit)
            else:
                log_data["Increase (%)"] = 100 * (self.data_misfit / self.prev_data_misfit - 1)
        else:
            log_data["Reduction (%)"] = 'N/A'

        self.logger(**log_data)

    def _ext_iter_param(self):
        """
        Extract parameters needed in LM-EnRML from the ITERATION keyword given in the DATAASSIM part of PIPT init.
        file. These parameters include convergence tolerances and parameters for the damping parameter. Default
        values for these parameters have been given here, if they are not provided in ITERATION.
        """

        # Predefine all the default values
        self.data_misfit_tol = 0.01
        self.step_tol = 0.01
        self.lam = 1.0
        #self.lam_max = 1e10
        #self.lam_min = 0.01
        self.gamma = 2
        self.trunc_energy = 0.95
        self.iteration = 0

        # Loop over options in ITERATION and extract the parameters we want
        for i, opt in enumerate(list(zip(*self.keys_da['iteration']))[0]):
            if opt == 'data_misfit_tol':
                self.data_misfit_tol = self.keys_da['iteration'][i][1]
            if opt == 'step_tol':
                self.step_tol = self.keys_da['iteration'][i][1]
            if opt == 'lambda':
                self.lam = self.keys_da['iteration'][i][1]
            if opt == 'lambda_max':
                self.lam_max = self.keys_da['iteration'][i][1]
            if opt == 'lambda_min':
                self.lam_min = self.keys_da['iteration'][i][1]
            if opt == 'lambda_factor':
                self.gamma = self.keys_da['iteration'][i][1]

        if 'energy' in self.keys_da:
            # initial energy (Remember to extract this)
            self.trunc_energy = self.keys_da['energy']
            if self.trunc_energy > 1:  # ensure that it is given as percentage
                self.trunc_energy /= 100.




