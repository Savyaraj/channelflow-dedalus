"""Lugiato-Lefever Equation (LLE) for a driven-dissipative photonic microresonator."""

from dedalus_interface import DedalusInterface
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import math
from dedalus import public as de
import h5py
import sys
import logging
import datetime
import gc
import copy
from scipy.io import loadmat
from scipy.io import savemat
import netCDF4 as nc

logger = logging.getLogger(__name__)


class DedalusPy(DedalusInterface):

    def __init__(self):

        self.dt = 1e-2
        self.N = 256
        self.N_Lc = 1

        self.parameters = {
            "z_0"    : 3,
            "d_2"    : 0.05,
            "f"      : 2,
            "a"      : 1,
            "z_0_end": 3.1,
            "I_0"    : 1,
        }

        if len(sys.argv) > 1:
            self.parameters["z_0"] = float(sys.argv[1])
            self.parameters["f"] = float(np.sqrt(float(sys.argv[2])))

        self.compute_background()
        self.parameters["z_"] = (self.parameters["a"] * self.parameters["z_0"] +
                                  (1 - self.parameters["a"]) * self.parameters["I_0"])
        print(self.parameters)

        self.discrete_sym = "none"
        self.ramp = 0
        self.init_field = "soliton"

        self.perturb = 0
        self.read_perturb = 0
        self.mag_perturb = 1e-3
        self.read_perturb_filename = "ef1.nc"

        self.read = 0
        self.read_niter = -1
        self.read_filename = "uinit.nc"
        self.read_h5_dir = "./"

        self.write = 0
        self.write_h5 = 0
        self.save_nc = 0
        self.save_mat = 0
        self.save_fig = 0
        self.write_fraction = 1
        self.write_filename = "./"
        self.mat_name = "LLE"
        self.write_iter = 10

        self.write_cont_mat = 0
        self.write_cont_mat_filename = "./mat/"

        self.Niter = 10000
        self.timestepper = de.RK222

        self.domain_setup(self.N_Lc)
        self.problem_setup(self.parameters["z_0"])
        self.build_solver(self.Niter)
        self.init_problem()
        super().__init__()
        self.save_info()

    def compute_background(self):
        zeta = self.parameters["z_0"]
        f = self.parameters["f"]
        a = self.parameters["a"]
        roots = np.roots([1, -2*zeta, 1 + zeta**2, -f**2])
        I_0 = np.min(roots[np.isreal(roots)])
        psi_0 = f / (1 + 1j*(zeta - I_0))
        self.parameters["I_0"] = I_0
        self.parameters["psi_0"] = psi_0

    def save_info(self, filename='./dedargs.asc'):
        with open(filename, 'a+') as f:
            f.write(f"\n\nLLE.py\t{datetime.datetime.now()}\n")
            f.write(f"dt\t{self.dt}\tN\t{self.N}\tN_Lc\t{self.N_Lc}\n")
            print(self.parameters, file=f)
            f.write(f"discrete_sym\t{self.discrete_sym}\n")
            f.write(f"read\t{self.read}\tread_iter\t{self.read_niter}\tread_filename\t{self.read_filename}\n")
            f.write(f"write\t{self.write}\twrite_h5\t{self.write_h5}\twrite_filename\t{self.write_filename}\twrite_iter\t{self.write_iter}\tNiter\t{self.Niter}\n")

    def domain_setup(self, N_Lc):
        self.Nphi = self.N
        self.L = 2*np.pi
        self.Lphi = N_Lc * self.L
        self.scale = 2  # dealias cubic nonlinearity

        self.phicoord = de.Coordinate('phi')
        self.dist = de.Distributor(self.phicoord, dtype=np.complex128)
        self.phi_basis = de.ComplexFourier(self.phicoord, self.Nphi,
                                           bounds=(-self.Lphi/2, self.Lphi/2),
                                           dealias=self.scale)
        self.phi = self.dist.local_grid(self.phi_basis)
        self.kphi = self.phi_basis.native_wavenumbers

    def ramp_function(self, z_0, z_end, t):
        if self.ramp:
            return z_0 + (z_end - z_0) * t / (self.dt * self.Niter)
        return z_0

    def problem_setup(self, mu=-1, muName='z_0'):
        self.psi = self.dist.Field(name='psi', bases=self.phi_basis)
        self.t = self.dist.Field(name='t')

        self.variables = {"psi": self.psi, "t": self.t}

        self.parameters[muName] = mu
        self.parameters["z_"] = (self.parameters["a"] * self.parameters["z_0"] +
                                  (1 - self.parameters["a"]) * self.parameters["I_0"])

        self.dphi = lambda psi: de.Differentiate(psi, self.phicoord)
        self.norm = lambda psi: psi * np.conj(psi)
        self.exp  = lambda x: np.exp(x)
        self.functions = {"dphi": self.dphi, "norm": self.norm,
                          "ramp": self.ramp_function, "exp": self.exp}

        self.namespace = {key: val
                          for d in [self.parameters, self.variables, self.functions]
                          for key, val in d.items()}

        self.problem = de.IVP([self.psi], time=self.t, namespace=self.namespace)
        self.problem.add_equation(
            "dt(psi) + psi - 1j*d_2*dphi(dphi(psi)) = "
            "-1j*ramp(z_0, z_0_end, t)*psi + 1j*norm(psi)*psi + f"
        )

    def build_solver(self, n_iter=1000):
        self.solver = self.problem.build_solver(self.timestepper)
        self.solver.stop_wall_time = np.inf
        self.solver.stop_iteration = n_iter

    def init_problem(self):
        self.psi.change_scales(1)

        if self.read:
            ext = self.read_filename.split(".")[-1]
            if ext == 'nc':
                u = nc.Dataset(self.read_filename, mode='r+')
                x = np.concatenate(
                    [u[field][:] for field in u.variables.keys() if field.startswith('Field_')],
                    axis=0)
                self.psi.change_scales(len(x) / (2 * self.N))
                self.psi['g'] = x[:len(x)//2] + 1j * x[len(x)//2:]
            elif ext == 'npy':
                psi_ = np.load(self.read_filename)
                self.psi.change_scales(int(len(psi_) / self.N))
                self.psi['g'] = psi_[:, 1] + 1j * psi_[:, 2]
            elif ext == 'mat':
                data = loadmat(self.read_filename)
                psi_data = np.array(data['psi'])
                self.psi['g'] = psi_data[0, :]
            else:
                h5path = self.read_h5_dir + self.read_filename + '/' + self.read_filename + '_s1.h5'
                with h5py.File(h5path, mode='r+') as file:
                    self.psi.change_scales(len(file['tasks']['psi'][self.read_niter]) / self.Nphi)
                    self.psi['g'] = file['tasks']['psi'][self.read_niter]
        else:
            self.init_state()

        if self.perturb:
            if self.read_perturb:
                up = nc.Dataset(self.read_perturb_filename, mode='r+')
                xp = np.concatenate(
                    [up[field][:] for field in up.variables.keys() if field.startswith('Field_')],
                    axis=0)
                psi_p = xp[:self.Nphi] + 1j * xp[self.Nphi:]
                self.psi['g'] += self.mag_perturb * (psi_p / np.linalg.norm(psi_p))
            else:
                self.psi['g'] += self.mag_perturb * (
                    np.random.rand(self.Nphi) + 1j * np.random.rand(self.Nphi))

        if self.write_h5:
            data = self.solver.evaluator.add_file_handler(self.write_filename, iter=self.write_iter)
            data.add_tasks(self.solver.state, layout='g')

    def init_state(self):
        if self.init_field == "random":
            np.random.seed(42)
            self.psi['g'] = (self.mag_perturb * np.random.rand(self.Nphi) +
                             1j * self.mag_perturb * np.random.rand(self.Nphi))
        elif self.init_field == "soliton":
            A = np.sqrt(2 * self.parameters["z_0"])
            phase = np.arccos(2 * A / (self.parameters["f"] * np.pi))
            self.psi['g'] =  A * np.exp(1j * phase) / np.cosh(A * self.phi / np.sqrt(2 * self.parameters["d_2"]))
        elif self.init_field == "const":
            self.psi['g'] = self.parameters["psi_0"]

    @property
    def system_name(self):
        return "LLE"

    def _get_initial_param(self):
        return self.parameters["z_0"]

    def _get_vector_size(self):
        return 2 * self.N

    def _get_default_mu_name(self):
        return "z_0"

    def shifts(self, a):
        self.psi['c'] = self.psi['c'] * np.exp(self.kphi * 1j * a)

    def phase_shifts(self, a):
        self.psi['g'] = self.psi['g'] * np.exp(1j * a)

    def to_field(self, x):
        self.psi.change_scales(1)
        for n in range(self.N):
            self.psi['g'][n] = x[n] + 1j * x[self.N + n]

    def to_vector(self):
        self.psi.change_scales(1)
        u_out = np.zeros(2 * self.N)
        for n in range(self.N):
            u_out[n] = self.psi['g'][n].real
            u_out[self.N + n] = self.psi['g'][n].imag
        return u_out

    def add_perturbations(self, mag, decay):
        self.psi['g'] = np.random.rand(self.Nphi) + 1j * np.random.rand(self.Nphi)
        self.psi['g'] /= np.linalg.norm(self.psi['g'])
        self.symmetry()
        return self.to_vector()

    def observable(self, x):
        self.to_field(x)
        return np.linalg.norm(self.psi['g']-self.parameters["psi_0"])**2 / self.N 

    def diff(self, u_init, dir):
        self.to_field(u_init)
        psi_diff = de.Differentiate(self.psi, self.phicoord).evaluate()
        self.psi['g'] = psi_diff['g']
        print(f"diff in direction {dir}: {np.linalg.norm(self.psi['g'])}\n")
        return self.to_vector()

    def advance(self, T, u_init=None):
        init_time = self.solver.sim_time
        self.psi.change_scales(1)

        if u_init is not None:
            if len(u_init) == 2 * self.N:
                self.to_field(u_init)
            else:
                print(f"warning: size mismatch (expected {2*self.N}, got {len(u_init)})\n")

        if not math.isfinite(T):
            return u_init

        step_size = min(self.dt, T)
        psi_array = []
        while self.solver.sim_time + step_size < init_time + T:
            self.solver.step(step_size)
            self.symmetry()
            self.psi.change_scales(1)
            psi_array.append(self.psi['g'].copy())
        gc.collect()

        self.solver.step(T - (self.solver.sim_time - init_time))
        self.symmetry()
        self.psi.change_scales(1)

        u_out = self.to_vector()
        u_out = (u_out - u_init[:2*self.N]) / T

        return u_out

    def updateMu(self, mu, muName):
        if muName == "None":
            muName = "z_0"

        if muName == 'L':
            self.domain_setup(mu)
            self.problem_setup(self.parameters["z_0"])
            self.build_solver(self.Niter)
            self.init_problem()
            print(f"updating L/Lc to: {mu}")
        elif muName in self.parameters:
            self.parameters[muName] = mu
            self.compute_background()
            self.problem_setup(mu, muName)
            self.build_solver(self.Niter)
            self.init_problem()
            print(f"updating {muName} to: {mu}")
        else:
            raise Exception(f"Incorrect mu name: {muName}\n")

        gc.collect()

    def read_h5(self, filename, iter):
        h5path = self.read_h5_dir + filename + '/' + filename + '_s1.h5'
        with h5py.File(h5path, mode='r+') as file:
            len_read = len(file['tasks']['psi'][iter])
            self.psi.change_scales(len_read / self.N)
            self.psi['g'] = file['tasks']['psi'][iter]
            return self.to_vector()

    def symmetry(self):
        if self.discrete_sym == 'even':
            self.psi.change_scales(1)
            for n in range(int(self.N/2)):
                self.psi['g'][n] = self.psi['g'][self.N - n - 1]
        elif self.discrete_sym == 'odd':
            self.psi.set_scales(1)
            for n in range(int(self.N/2)):
                self.psi['g'][n] = -self.psi['g'][self.N - n - 1]
        elif self.discrete_sym == 'const':
            self.psi.change_scales(1)
            self.psi['c'][self.kphi != 0] = 0

if __name__ == "__main__":

    Dd = DedalusPy()

    Dd.psi.change_scales(1)
    psi_list      = [np.copy(np.abs(Dd.psi['g']))]
    psi_full_list = [np.copy(Dd.psi['g'])]
    psi_norm_list = [np.linalg.norm(Dd.psi['g'])**2 / Dd.N]
    t_list        = [Dd.solver.sim_time]

    savepath = (Dd.write_filename + "/plots/N_" + str(Dd.N)
                + "_f_" + str(round(Dd.parameters["f"], 5))
                + "_z_0_" + str(round(Dd.parameters["z_0"], 5)))

    if Dd.write_h5:
        Dd.save_info(Dd.write_filename + '/dedargs.asc')

    try:
        start_time = time.time()
        while Dd.solver.proceed:
            Dd.solver.step(Dd.dt)

            if Dd.solver.iteration % Dd.write_iter == 0:
                Dd.psi.change_scales(1)
                psi_list.append(np.copy(np.abs(Dd.psi['g'])))
                psi_full_list.append(np.copy(Dd.psi['g']))
                psi_norm_list.append(np.linalg.norm(Dd.psi['g'])**2 / Dd.N)
                t_list.append(Dd.solver.sim_time)

            if Dd.solver.iteration % (10 * Dd.write_iter) == 0:
                obs = round(Dd.observable(Dd.to_vector()), 5)
                print(f"Completed iteration {Dd.solver.iteration}, 2-norm {obs}")

        psi_array      = np.array(psi_list)
        psi_full_array = np.array(psi_full_list)
        psi_norm_array = np.array(psi_norm_list)
        t_array        = np.array(t_list)

        n_keep = int(Dd.write_fraction * len(t_array))
        psi_array      = psi_array[-n_keep:]
        psi_full_array = psi_full_array[-n_keep:]
        psi_norm_array = psi_norm_array[-n_keep:]
        t_array        = t_array[-n_keep:]

        if Dd.save_mat:
            fname = (Dd.write_filename
                     + f"f_2_{round(Dd.parameters['f']**2, 3)}"
                     + f"_z_0_{round(Dd.parameters['z_0'], 3)}"
                     + f"_L_{round(Dd.Lphi, 3)}_{Dd.mat_name}.mat")
            savemat(fname, {"psi": psi_full_array, "t": t_array, "x": Dd.phi})

        if Dd.save_nc:
            u = nc.Dataset('uinit.nc', mode='r+')
            u['Field_0'][:] = Dd.to_vector()
            u.close()

        fig, ax = plt.subplots(1, 2)
        fig1 = ax[0].pcolormesh(Dd.phi, t_array, psi_array, shading='nearest', cmap='inferno')
        fig.colorbar(fig1)
        ax[0].set_xlabel(r'$\phi$')
        ax[0].set_ylabel('t')
        ax[0].set_title(r'$|\psi|$')

        psi_imag_array = np.linalg.norm(np.imag(psi_full_array), axis=1)**2 / Dd.N
        ax[1].plot(t_array, psi_imag_array - psi_imag_array[0])
        ax[1].set_xlabel(r'$t$')
        ax[1].set_ylabel(r'$\|\mathrm{Im}(\psi)\|_2^2$')
        ax[1].yaxis.tick_right()

        fig.tight_layout()
        if Dd.write:
            plt.savefig(savepath + "/spatiotemporal_dynamics.png", dpi=1200)
        elif Dd.write_h5 or Dd.save_fig:
            plt.savefig(Dd.write_filename + "/spatiotemporal_dynamics.png", dpi=1200)
        else:
            plt.savefig("spatiotemporal_dynamics.png", dpi=1200)
        plt.show()

    except Exception:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()
        logger.info('Iterations: %i', Dd.solver.iteration)
        logger.info('Sim end time: %f', Dd.solver.sim_time)
        logger.info('Run time: %.2f sec', end_time - start_time)
        logger.info('Run time: %f cpu-hr', (end_time - start_time) / 3600 * Dd.dist.comm_cart.size)
