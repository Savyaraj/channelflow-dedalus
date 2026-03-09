""" Dedalus-nsolver interface class for specifying dynamical systems in dedalus

    The setting up PDEs in dedalus for timestepping, computing observables (norms),
    updating parameters and communications with nsolver library in C++ are specified
    here

"""

from abc import ABC, abstractmethod
import datetime

class DedalusInterface(ABC):

    def __init__(self):
        """Template method for initialization.

        Subclasses must set configuration attributes (dt, N, parameters, Niter,
        timestepper, N_Lc) BEFORE calling super().__init__().
        """
        # Validate required attributes
        self._validate_config()

        # Execute setup chain
        self.domain_setup(self.N_Lc)
        self.problem_setup(self._get_initial_param())
        self.build_solver(self.Niter)
        self.init_problem()

        # Post-initialization
        print(f"-----------{self.system_name} problem setup complete--------------\n")
        self.save_info()

    def _validate_config(self):
        """Ensure subclass set required attributes before calling super().__init__()"""
        required = ['dt', 'N', 'parameters', 'Niter', 'timestepper', 'N_Lc']
        for attr in required:
            if not hasattr(self, attr):
                raise AttributeError(f"Subclass must set '{attr}' before calling super().__init__()")

    @property
    @abstractmethod
    def system_name(self):
        """Return system name for logging (e.g., 'LLE_laser')"""
        pass

    @abstractmethod
    def _get_initial_param(self):
        """Return initial value of primary continuation parameter"""
        pass

    def save_info(self, filename='./dedargs.asc'):
        """Template method for logging configuration"""
        with open(filename, 'a+') as f:
            f.write(f"\n\n{self.system_name}.py\t{datetime.datetime.now()}\n")
            f.write(f"dt\t{self.dt}\tN\t{self.N}\tN_Lc\t{self.N_Lc}\n")
            print(self.parameters, file=f)
            self._save_info_hook(f)

    def _save_info_hook(self, f):
        """Hook for system-specific logging (optional override)"""
        pass

    def domain_setup(self):
        """Setup dedalus domain/grid.

        Can accept arguments to update domain size during continuations
        """
        pass 

    def problem_setup(self):
        """Setup the variables, equations and boundary conditions.
        
        Can also accept paramaters to update the system during continuations
        """
        pass 

    def build_solver(self):
        """Build the dedalus solver and set timesteppers, simulation time/number of iterations"""
        pass 
    
    def init_problem(self):
        """Intialize the system in dedalus"""
        pass 

    def shifts(self):
        """Shift operations on system state for special solutions."""
        pass 

    @abstractmethod
    def add_perturbations(self, mag, decay):
        """Create perturbation fields in order to compute eigenvalue spectrum.
        
        Returns perturbation as nsolver vector of the same size as field variables
        It must be verified that the perturbed field satisfies additional constraints and
        boundary conditions 
        Keyword arguments:
        mag   -> perturbation magnitude, given as 2-norm of the field
        decay -> decay in spectral perturbations for larger wavenumbers (default = 1)
        """
        pass 
        
    @abstractmethod
    def observable(self, x):
        """Observable of the system state, used for performing continuation.
        
        Returns a scalar of interest, such as L2norm, dissipation, energy, etc.
        Keyword arguments:
        x -> nsolver vector 
        """
        pass 
        
    @abstractmethod
    def diff(self, x, dir):
        """Compute partial derivatives of the fields in a given direction.

        Can be employed to satisfy orthogonality constraints in nsolver for
        for additional unknowns such as speeds of travelling solutions.
        Returns back an nsolver vector of same length as x 
        Keyword arguments:
        x   -> nsolver vector
        dir -> axis for partial differentiation, currently 'x' and 'y' are 
        the only valid arguments
        """
        pass 

    def advance(self, T, u_init=None):
        """Template method for time integration.

        Used by nsolver to approximate the Jacobians by finite differences.
        Returns (u_final - u_init)/T as nsolver vector.

        Keyword arguments:
        T      -> time of integration
        u_init -> nsolver vector (initial state)
        """
        import math
        init_time = self.solver.sim_time

        # Input validation & conversion
        if u_init is not None:
            if len(u_init) == self._get_vector_size():
                self.to_field(u_init)
            else:
                print(f"warning: size mismatch (expected {self._get_vector_size()}, got {len(u_init)})\n")

        if not math.isfinite(T):
            return u_init if u_init is not None else self.to_vector()

        # Timestep loop
        step_size = min(self.dt, T)
        while self.solver.sim_time + step_size < init_time + T:
            self.solver.step(step_size)
            self.symmetry()

        # Final step to reach exactly T
        self.solver.step(T - (self.solver.sim_time - init_time))
        self.symmetry()

        # Convert to vector
        self._change_scales_hook(1)
        u_out = self.to_vector()

        # Compute derivative
        if u_init is not None:
            u_out = (u_out - u_init) / T

        return u_out

    @abstractmethod
    def _get_vector_size(self):
        """Return size of flattened nsolver vector"""
        pass

    def _change_scales_hook(self, scale):
        """Hook to change field scales (override if needed)"""
        pass 
        
    def updateMu(self, mu, muName):
        """Template method for parameter updates.

        For updating system parameters, dedalus solver must be reconstructed at each
        continuation step. If continuation in domain size is desired, the dedalus
        domain must be reconstructed along with the solver.

        Keyword argments:
        mu     -> parameter value
        muName -> parameter name, given as a string
        """
        if muName == "None":
            muName = self._get_default_mu_name()

        if muName == 'L':
            # Domain size change
            self.domain_setup(mu)
            self.problem_setup(self._get_current_param_value())
            self.build_solver(self.Niter)
            self.init_problem()
            print(f"updating L/Lc to: {mu}")
        elif muName in self.parameters:
            # Parameter change
            self.parameters[muName] = mu
            self.domain_setup(self.N_Lc)
            self.problem_setup(mu, muName)
            self.build_solver(self.Niter)
            self.init_problem()
            print(f"updating {muName} to: {mu}")
        else:
            raise Exception(f"Incorrect mu name: {muName}\n")

    @abstractmethod
    def _get_default_mu_name(self):
        """Return default parameter name (e.g., 'a_L', 'a')"""
        pass

    def _get_current_param_value(self):
        """Get current value of primary parameter"""
        return self.parameters[self._get_default_mu_name()] 
    
    @abstractmethod
    def read_h5(self, filename, iter):
        """Interface to read from h5 files in nsolver.

        Returns an nsolver vector with the state loaded from h5 file
        Keyword arguments:
        filename -> name of the h5 file, given as a string
        iter     -> iteration number
        """
        pass 
        
    def symmetry(self):
        """Perform symmetry operations on the state ."""
        pass 
        