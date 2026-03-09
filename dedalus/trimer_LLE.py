"""Linearly coupled LLE equations for a driven-dissipative photonic trimer.

Based on: 'Emergent Nonlinear Phenomena in Driven Dissipative Photonic Dimer',
Tikan et al. (2020).
"""

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
import netCDF4 as nc

logger = logging.getLogger(__name__)


class DedalusPy(DedalusInterface):

    def __init__(self):

        self.dt = 1e-3
        self.N = 512
        self.N_Lc = 1

        self.parameters = {
            "z_0"    : 8,
            "d_2"    : 0.04,
            "f"      : 8,
            "J"      : 10,
            "z_0_end": 10,
        }

        if len(sys.argv) > 1:
            self.parameters["J"] = float(sys.argv[1])
            self.parameters["z_0"] = -2 * self.parameters["J"]
            self.parameters["z_0_end"] = 2 * self.parameters["J"]
            if len(sys.argv) > 2:
                self.parameters["f"] = float(sys.argv[2])

        print(self.parameters)

        self.discrete_sym = "none"
        self.ramp = 0
        self.init_field = "random"

        self.perturb = 0
        self.read_perturb = 0
        self.mag_perturb = 1e-1
        self.read_perturb_dir = "./"

        self.read = 0
        self.read_niter = -1
        self.read_filename = "ubest.nc"
        self.read_h5_dir = "./"
        self.read_shift = 0

        self.write = 0
        self.write_h5 = 0
        self.write_filename = ("dedalus_data/trimer/N_" + str(self.N)
                               + "_J_" + str(round(self.parameters["J"], 5))
                               + "_f_" + str(round(self.parameters["f"], 5))
                               + "_z_0_" + str(round(self.parameters["z_0"], 5)))
        self.write_filename = self.write_filename.replace(".", "-")
        if not os.path.isdir(self.write_filename) and self.write_h5:
            os.makedirs(self.write_filename)
        self.write_iter = 10
        self.Niter = 1000
        self.timestepper = de.RK222

        self.domain_setup(self.N_Lc)
        self.problem_setup(self.parameters["z_0"])
        self.build_solver(self.Niter)
        self.init_problem()
        super().__init__()
        self.save_info()

    def save_info(self, filename='./dedargs.asc'):
        with open(filename, 'a+') as f:
            f.write(f"\n\ntrimer_LLE.py\t{datetime.datetime.now()}\n")
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
        self.psi_1 = self.dist.Field(name='psi_1', bases=self.phi_basis)
        self.psi_2 = self.dist.Field(name='psi_2', bases=self.phi_basis)
        self.psi_3 = self.dist.Field(name='psi_3', bases=self.phi_basis)
        self.t = self.dist.Field(name='t')

        self.variables = {"psi_1": self.psi_1, "psi_2": self.psi_2,
                          "psi_3": self.psi_3, "t": self.t}

        self.parameters[muName] = mu

        self.dphi = lambda psi: de.Differentiate(psi, self.phicoord)
        self.norm = lambda psi: psi * np.conj(psi)
        self.exp  = lambda x: np.exp(x)
        self.functions = {"dphi": self.dphi, "norm": self.norm,
                          "ramp": self.ramp_function, "exp": self.exp}

        self.namespace = {key: val
                          for d in [self.parameters, self.variables, self.functions]
                          for key, val in d.items()}

        self.problem = de.IVP([self.psi_1, self.psi_2, self.psi_3], time=self.t, namespace=self.namespace)
        self.problem.add_equation(
            "dt(psi_1) + psi_1 - 1j*d_2*dphi(dphi(psi_1)) - 1j*J*psi_2 = "
            "-1j*ramp(z_0, z_0_end, t)*psi_1 + 1j*norm(psi_1)*psi_1 + f"
        )
        self.problem.add_equation(
            "dt(psi_2) + psi_2 - 1j*d_2*dphi(dphi(psi_2)) - 1j*J*psi_1 - 1j*J*psi_3 = "
            "-1j*ramp(z_0, z_0_end, t)*psi_2 + 1j*norm(psi_2)*psi_2"
        )
        self.problem.add_equation(
            "dt(psi_3) + psi_3 - 1j*d_2*dphi(dphi(psi_3)) - 1j*J*psi_2 = "
            "-1j*ramp(z_0, z_0_end, t)*psi_3 + 1j*norm(psi_3)*psi_3"
        )

    def build_solver(self, n_iter=1000):
        self.solver = self.problem.build_solver(self.timestepper)
        self.solver.stop_wall_time = np.inf
        self.solver.stop_iteration = n_iter

    def init_problem(self):
        self.psi_1.change_scales(1)
        self.psi_2.change_scales(1)
        self.psi_3.change_scales(1)

        if self.read:
            ext = self.read_filename.split(".")[-1]
            if ext == 'nc':
                u = nc.Dataset(self.read_filename, mode='r+')
                x = np.concatenate((u['Field_0'][:], u['Field_1'][:],
                                     u['Field_2'][:], u['Field_3'][:],
                                     u['Field_4'][:], u['Field_5'][:]), axis=0)
                self.to_field(x)
            else:
                h5path = self.read_h5_dir + self.read_filename + '/' + self.read_filename + '_s1.h5'
                with h5py.File(h5path, mode='r+') as file:
                    self.psi_1['g'] = file['tasks']['psi_1'][self.read_niter]
                    self.psi_2['g'] = file['tasks']['psi_2'][self.read_niter]
                    self.psi_3['g'] = file['tasks']['psi_3'][self.read_niter]
                    self.shifts(self.read_shift)
        else:
            self.init_state()

        if self.perturb:
            if self.read_perturb:
                ev_arr = np.load(self.read_perturb_dir + "eigenvectors.npy")
                self.psi_1['g'] += self.mag_perturb * ev_arr[5, 0, :] / np.linalg.norm(ev_arr[5, 0, :])
                self.psi_2['g'] += self.mag_perturb * ev_arr[5, 2, :] / np.linalg.norm(ev_arr[5, 2, :])
            else:
                self.psi_1['g'] += self.mag_perturb * (np.random.rand(self.Nphi) + 1j * np.random.rand(self.Nphi))
                self.psi_2['g'] += self.mag_perturb * (np.random.rand(self.Nphi) + 1j * np.random.rand(self.Nphi))
                self.psi_3['g'] += self.mag_perturb * (np.random.rand(self.Nphi) + 1j * np.random.rand(self.Nphi))

        if self.write_h5:
            data = self.solver.evaluator.add_file_handler(self.write_filename, iter=self.write_iter)
            data.add_tasks(self.solver.state, layout='g')

    def init_state(self):
        if self.init_field == "random":
            self.psi_1['g'] = np.random.rand(self.Nphi)
            self.psi_2['g'] = np.random.rand(self.Nphi)
            self.psi_3['g'] = np.random.rand(self.Nphi)
        elif self.init_field == "soliton":
            self.psi_1['g'] = 3.5 * 1j / np.cosh(4 * self.phi / np.sqrt(self.parameters["d_2"]))
        elif self.init_field == "const":
            self.psi_1['g'] = 0.01 * np.ones(self.Nphi)
            self.psi_2['g'] = 0.01 * np.ones(self.Nphi)
            self.psi_3['g'] = 0.01 * np.ones(self.Nphi)

    @property
    def system_name(self):
        return "trimer_LLE"

    def _get_initial_param(self):
        return self.parameters["z_0"]

    def _get_vector_size(self):
        return 6 * self.N

    def _get_default_mu_name(self):
        return "z_0"

    def shifts(self, a):
        self.psi_1['c'] = self.psi_1['c'] * np.exp(self.kphi * 1j * a)
        self.psi_2['c'] = self.psi_2['c'] * np.exp(self.kphi * 1j * a)
        self.psi_3['c'] = self.psi_3['c'] * np.exp(self.kphi * 1j * a)

    def to_field(self, x):
        self.psi_1.change_scales(1)
        self.psi_2.change_scales(1)
        self.psi_3.change_scales(1)
        for n in range(self.N):
            self.psi_1['g'][n] = x[n] + 1j * x[self.N + n]
            self.psi_2['g'][n] = x[2*self.N + n] + 1j * x[3*self.N + n]
            self.psi_3['g'][n] = x[4*self.N + n] + 1j * x[5*self.N + n]

    def to_vector(self):
        self.psi_1.change_scales(1)
        self.psi_2.change_scales(1)
        self.psi_3.change_scales(1)
        u_out = np.zeros(6 * self.N)
        for n in range(self.N):
            u_out[n]           = self.psi_1['g'][n].real
            u_out[self.N + n]  = self.psi_1['g'][n].imag
            u_out[2*self.N + n] = self.psi_2['g'][n].real
            u_out[3*self.N + n] = self.psi_2['g'][n].imag
            u_out[4*self.N + n] = self.psi_3['g'][n].real
            u_out[5*self.N + n] = self.psi_3['g'][n].imag
        return u_out

    def add_perturbations(self, mag, decay):
        self.psi_1['g'] = np.random.rand(self.Nphi) + 1j * np.random.rand(self.Nphi)
        self.psi_2['g'] = np.random.rand(self.Nphi) + 1j * np.random.rand(self.Nphi)
        self.psi_3['g'] = np.random.rand(self.Nphi) + 1j * np.random.rand(self.Nphi)
        self.psi_1['g'] /= np.linalg.norm(self.psi_1['g'])
        self.psi_2['g'] /= np.linalg.norm(self.psi_2['g'])
        self.psi_3['g'] /= np.linalg.norm(self.psi_3['g'])
        self.symmetry()
        return self.to_vector()

    def observable(self, x):
        self.to_field(x)
        return np.linalg.norm(self.psi_1['g']) / np.sqrt(self.N)

    def diff(self, u_init, dir):
        self.to_field(u_init)
        psi_1_diff = de.Differentiate(self.psi_1, self.phicoord).evaluate()
        psi_2_diff = de.Differentiate(self.psi_2, self.phicoord).evaluate()
        psi_3_diff = de.Differentiate(self.psi_3, self.phicoord).evaluate()
        self.psi_1['g'] = psi_1_diff['g']
        self.psi_2['g'] = psi_2_diff['g']
        self.psi_3['g'] = psi_3_diff['g']
        print(f"diff in direction {dir}: {np.linalg.norm(self.psi_1['g'])}\n")
        return self.to_vector()

    def advance(self, T, u_init=None):
        init_time = self.solver.sim_time
        self.psi_1.change_scales(1)
        self.psi_2.change_scales(1)
        self.psi_3.change_scales(1)

        if u_init is not None:
            if len(u_init) == 6 * self.N:
                self.to_field(u_init)
            else:
                print("warning: size mismatch from python API\n")

        if not math.isfinite(T):
            return u_init

        step_size = min(self.dt, T)
        while self.solver.sim_time + step_size < init_time + T:
            self.solver.step(step_size)
            self.symmetry()
        gc.collect()

        self.solver.step(T - (self.solver.sim_time - init_time))
        self.symmetry()
        self.psi_1.change_scales(1)
        self.psi_2.change_scales(1)
        self.psi_3.change_scales(1)

        u_out = self.to_vector()
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
            self.psi_1['g'] = file['tasks']['psi_1'][iter]
            self.psi_2['g'] = file['tasks']['psi_2'][iter]
            self.psi_3['g'] = file['tasks']['psi_3'][iter]
            return self.to_vector()

    def symmetry(self):
        if self.discrete_sym == 'even':
            self.psi_1.change_scales(1)
            self.psi_2.change_scales(1)
            self.psi_3.change_scales(1)
            for n in range(int(self.N/2)):
                self.psi_1['g'][n] = self.psi_1['g'][self.N - n - 1]
                self.psi_2['g'][n] = self.psi_2['g'][self.N - n - 1]
                self.psi_3['g'][n] = self.psi_3['g'][self.N - n - 1]
        elif self.discrete_sym == 'const':
            self.psi_1.change_scales(1)
            self.psi_2.change_scales(1)
            self.psi_3.change_scales(1)
            self.psi_1['c'][self.kphi != 0] = 0
            self.psi_2['c'][self.kphi != 0] = 0
            self.psi_3['c'][self.kphi != 0] = 0


if __name__ == "__main__":

    Dd = DedalusPy()

    Dd.psi_1.change_scales(1)
    Dd.psi_2.change_scales(1)
    Dd.psi_3.change_scales(1)
    psi_1_list = [np.copy(np.abs(Dd.psi_1['g']))]
    psi_2_list = [np.copy(np.abs(Dd.psi_2['g']))]
    psi_3_list = [np.copy(np.abs(Dd.psi_3['g']))]
    psi_1_norm_list = [np.linalg.norm(np.abs(Dd.psi_1['g'])) / np.sqrt(Dd.N)]
    psi_2_norm_list = [np.linalg.norm(np.abs(Dd.psi_2['g'])) / np.sqrt(Dd.N)]
    psi_3_norm_list = [np.linalg.norm(np.abs(Dd.psi_3['g'])) / np.sqrt(Dd.N)]
    t_list = [Dd.solver.sim_time]

    if Dd.write_h5:
        Dd.save_info(Dd.write_filename + '/dedargs.asc')

    try:
        start_time = time.time()
        while Dd.solver.proceed:
            Dd.solver.step(Dd.dt)
            Dd.symmetry()

            if Dd.solver.iteration % Dd.write_iter == 0:
                Dd.psi_1.change_scales(1)
                Dd.psi_2.change_scales(1)
                Dd.psi_3.change_scales(1)
                psi_1_list.append(np.copy(np.abs(Dd.psi_1['g'])))
                psi_2_list.append(np.copy(np.abs(Dd.psi_2['g'])))
                psi_3_list.append(np.copy(np.abs(Dd.psi_3['g'])))
                psi_1_norm_list.append(np.linalg.norm(np.abs(Dd.psi_1['g'])) / np.sqrt(Dd.N))
                psi_2_norm_list.append(np.linalg.norm(np.abs(Dd.psi_2['g'])) / np.sqrt(Dd.N))
                psi_3_norm_list.append(np.linalg.norm(np.abs(Dd.psi_3['g'])) / np.sqrt(Dd.N))
                t_list.append(Dd.solver.sim_time)

            if Dd.solver.iteration % (10 * Dd.write_iter) == 0:
                n1 = round(np.linalg.norm(np.abs(Dd.psi_1['g'])) / np.sqrt(Dd.N), 5)
                n2 = round(np.linalg.norm(np.abs(Dd.psi_2['g'])) / np.sqrt(Dd.N), 5)
                n3 = round(np.linalg.norm(np.abs(Dd.psi_3['g'])) / np.sqrt(Dd.N), 5)
                print(f"Completed iteration {Dd.solver.iteration}, 2-norm cavity 1 {n1}, cavity 2 {n2}, cavity 3 {n3}")

        psi_1_array = np.array(psi_1_list)
        psi_2_array = np.array(psi_2_list)
        psi_3_array = np.array(psi_3_list)
        t_array = np.array(t_list)

        tick_pos = [-np.pi, 0, np.pi]
        labels   = [r'$-\pi$', '0', r'$\pi$']

        fig, ax = plt.subplots(1, 4)
        ax[0].pcolormesh(Dd.phi, t_array, psi_1_array, shading='nearest')
        ax[0].set_xlabel(r'$\phi$')
        ax[0].set_xticks(tick_pos, labels)
        ax[0].set_ylabel('t')
        ax[0].set_title(r'$|\psi_1|$')

        ax[1].pcolormesh(Dd.phi, t_array, psi_2_array, shading='nearest')
        ax[1].set_xlabel(r'$\phi$')
        ax[1].set_xticks(tick_pos, labels)
        ax[1].get_yaxis().set_visible(False)
        ax[1].set_title(r'$|\psi_2|$')

        ax[2].pcolormesh(Dd.phi, t_array, psi_3_array, shading='nearest')
        ax[2].set_xlabel(r'$\phi$')
        ax[2].set_xticks(tick_pos, labels)
        ax[2].get_yaxis().set_visible(False)
        ax[2].set_title(r'$|\psi_3|$')

        if Dd.ramp:
            z_vals = np.linspace(Dd.parameters['z_0'], Dd.parameters["z_0_end"], len(psi_1_norm_list))
            ax[3].plot(psi_1_norm_list, z_vals, label=r'$\psi_1$')
            ax[3].plot(psi_2_norm_list, z_vals, label=r'$\psi_2$')
            ax[3].plot(psi_3_norm_list, z_vals, label=r'$\psi_3$')
            ax[3].set_xlabel(r'$\|\psi\|_2$')
            ax[3].set_ylabel(r'$\zeta$')
        else:
            ax[3].plot(psi_1_norm_list, t_array, label=r'$\psi_1$')
            ax[3].plot(psi_2_norm_list, t_array, label=r'$\psi_2$')
            ax[3].plot(psi_3_norm_list, t_array, label=r'$\psi_3$')
            ax[3].set_xlabel(r'$\|\psi\|_2$')
            ax[3].set_ylabel(r'$t$')
        ax[3].yaxis.set_label_position("right")
        ax[3].yaxis.tick_right()
        ax[3].legend(loc="lower right", fontsize=7, markerscale=0.4, framealpha=0.4)

        fig.tight_layout()
        if Dd.write:
            savepath = (Dd.write_filename + "/plots/N_" + str(Dd.N)
                        + "_J_" + str(round(Dd.parameters["J"], 5))
                        + "_f_" + str(round(Dd.parameters["f"], 5))
                        + "_z_0_" + str(round(Dd.parameters["z_0"], 5)))
            plt.savefig(savepath + "/spatiotemporal_dynamics.png", dpi=1200)
        elif Dd.write_h5:
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
