/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */
#include <sys/stat.h>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "cfbasics/cfvector.h"
// #include "channelflow/cfdsi.h"
#include "channelflow/dedadsi.h"
#include "channelflow/dedafield.h"
// #include "channelflow/chebyshev.h"
// #include "channelflow/dns.h"
// #include "channelflow/flowfield.h"
// #include "channelflow/symmetry.h"
// #include "channelflow/tausolver.h"
#include "nsolver/nsolver.h"

using namespace std;
using namespace Eigen;
using namespace chflow;

// This program calculates eigenvalues of fixed point of plane Couette flow
// using Arnoldi iteration. The ideas and algorithm are based on Divakar
// Viswanath, "Recurrent motions within plane Couette turbulence",
// J. Fluid Mech.</em> 580 (2007), http://arxiv.org/abs/physics/0604062.

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    {
        // WriteProcessInfo(argc, argv);
        int taskid = 0;
#ifdef HAVE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#endif

        string purpose(
            "compute spectrum of eigenvalues of equilibria, traveling waves, or periodic orbit using Arnoldi "
            "iteration");
        ArgList args(argc, argv, purpose);

        // DNSFlags dnsflags(args);
        // TimeStep dt(dnsflags);
        // dnsflags.verbosity = Silent;

        // The Eigenvals class is utilized to solve the eigenvalue problem.
        // This class requires Arnoldi class.
        unique_ptr<Eigenvals> E;
        EigenvalsFlags eigenflags(args);
        E = unique_ptr<Eigenvals>(new Eigenvals(eigenflags));

        args.section("Program options");
        // const bool poincare =
        //     args.getflag("-poinc", "--poincare", "computing eigenvalues of map on I-D=0 Poincare section");

        // const string sigstr =
        //     args.getstr("-sigma", "--sigma", "", "file containing sigma of sigma f^T(u) - u = 0 (default == identity)");

        const int seed = args.getint("-sd", "--seed", 1, "seed for random number generator");
        const Real smooth = args.getreal("-s", "--smoothness", 0.4, "smoothness of initial perturb, 0 < s < 1");
        const Real EPS_du = args.getreal("-edu", "--epsdu", 1e-7,
                                         "magnitude of perturbation for numerical approximation of the Jacobian");
        const string duname = args.getstr("-du", "--perturb", "", "initial perturbation field, random if unset");

        // const int nproc0 =
        //     args.getint("-np0", "--nproc0", 0, "number of MPI-processes for transpose/number of parallel ffts");
        // const int nproc1 = args.getint("-np1", "--nproc1", 0, "number of MPI-processes for one fft");
        const bool mu_read = args.getflag("-mu_read", "--mu_read", "read parameter value");
        const string muname = args.getstr("-mu_name", "--mu_name",".", "parameter name");
        Real mu = args.getreal("-mu", "--Mu", 0.0, "Parameter value");
    
        const Real T = args.getreal("-T", "--T", "total time for integration");
        const string sys = args.getstr("-sys", "--system","active_matter", "system for time integration (python file name)");
        const string uname = args.getstr(1, "<flowfield>", "filename for EQB, TW, or PO solution");
        const Real eps_gx = args.getreal("-eps_gx", "--eps_gx", 1e-6, "tolerance for Arnoldi iteration");

        const bool Rxsearch = args.getflag("-xrel", "--xrelative", "x shifts for relative solution");
        const bool Rzsearch = args.getflag("-zrel", "--zrelative", "z shifts for relative solution");

        const string shiftsname = args.getstr("-shifts", "--shifts",".", "filename for shifts in travelling wave");
        // CfMPI* cfmpi = &CfMPI::getInstance(nproc0, nproc1);

        args.check();
        args.save("./");
        args.save(eigenflags.outdir);

        // fftw_loadwisdom();

        srand48(seed);
        const Real decay = 1.0 - smooth;

        // PoincareCondition* h = poincare ? new DragDissipation() : 0;

        DedaField u;  // u*, the solution of sigma f^T(u*) - u* = 0
        
        u.readNetCDF(uname);
        std::map<string, Real> shifts;
        std::map<string, Real> shifts_perturb;

        if (Rxsearch || Rzsearch){
        ifstream is;
        string filename = ifstreamOpen(is, shiftsname, ".asc");
        if (!is) {
            cerr << "shifts : can't open file " << shiftsname << " or " << (shiftsname + ".asc")
                << endl;
            exit(1);
        }

        // Read in header. Form is "%N a b s"
        string comment;
        while (is.peek() == '%')
            getline(is, comment);
        int num_shifts;
        is >> num_shifts;
        for (int i = 0; i < num_shifts; i++){
            std::string dim;
            Real a;
            is>>dim>>a;
            shifts[dim] = a;
            shifts_perturb[dim] = 0.0;
        }
        if (!is.good())
            cerr << "warning: bad istream in reading shifts from file " << filename << endl;
        is.close();
        }
        // const int Nx = u.Nx();
        // const int Ny = u.Ny();
        // const int Nz = u.Nz();

        // const int kxmin = -u.kxmaxDealiased();
        // const int kxmax = u.kxmaxDealiased();
        // const int kzmin = 0;
        // const int kzmax = u.kzmaxDealiased();

        // if (taskid == 0) {
        //     cout << setprecision(17);
        //     cout << "   Nx == " << Nx << endl;
        //     cout << "   Ny == " << Ny << endl;
        //     cout << "   Nz == " << Nz << endl;
        //     cout << "kxmin == " << kxmin << endl;
        //     cout << "kxmax == " << kxmax << endl;
        //     cout << "kzmin == " << kzmin << endl;
        //     cout << "kzmax == " << kzmax << endl;

        //     cout << "dt     == " << dt.dt() << endl;
        //     cout << "dtmin  == " << dt.dtmin() << endl;
        //     cout << "dtmax  == " << dt.dtmax() << endl;
        //     cout << "CFLmin == " << dt.CFLmin() << endl;
        //     cout << "CFLmax == " << dt.CFLmax() << endl;
        // }
        // if (!poincare)
        //     dt.adjust_for_T(dnsflags.T);

        // FieldSymmetry sigma;  // defaults to identity
        // if (sigstr.length() != 0)
        //     sigma = FieldSymmetry(sigstr);

        // if (dnsflags.symmetries.length() > 0) {
        //     if (taskid == 0) {
        //         cout << "Restricting flow to invariant subspace generated by symmetries" << endl;
        //         cout << dnsflags.symmetries << endl;
        //     }
        // }

        // if (taskid == 0)
        //     cout << "DNS flags = " << dnsflags << endl;
        // dnsflags.save(eigenflags.outdir);

        // Set up DNS operator ("A" in Arnoldi A*b terms)
        // if (taskid == 0)
        //     cout << "setting up DNS and initial fields..." << endl;
        // ChebyTransform trans(Ny);

        const Real l2u = u.L2Norm();
        const Real eps =
            (l2u < EPS_du)
                ? EPS_du
                : EPS_du / l2u;  // for choice of epsilon, see eq. (15) in reference
                                 // C.J. Mack, P.J. Schmid/Journal of Computational Physics 229 (2010) 541-560
        // DedaField Gu(u);
        if (taskid == 0)
            cout << "computing sigma f^T(u)..." << endl;

        // // Construct the dynamical-systems interface object depending on the given parameters.
        // unique_ptr<cfDSI> dsi;
        // dsi =
        //     unique_ptr<cfDSI>(new cfDSI(dnsflags, sigma, h, dt, false, false, false, false, 0.0, u, E->getLogstream()));

        unique_ptr<dedaDSI> dsi;
        dsi = unique_ptr<dedaDSI>(new dedaDSI(T, u, false, Rxsearch, Rzsearch, shifts, sys));

        if (mu_read){
            dsi->chooseMu(muname);
            dsi->updateMu(mu);
        }
        // Check if sigma f^T(u) - u = 0
        VectorXd x;
        u.toVector(x);
        // dsi->makeVector(u, shifts, T, x);

        VectorXd Gx = dsi->eval(x);

        // if (taskid == 0)
        //     cout << "\nCFL == " << dsi->getCFL() << endl;

        Real l2normGx = L2Norm(Gx);
        if (taskid == 0) {
            cout << "L2Norm(Gx = (x - sigma f^T(x)) ) = " << l2normGx << endl;
            cout << "L2Norm(Gx normalized = (x - sigma f^T(x))/T ) = " << l2normGx / T << endl;
        }

        if (l2normGx > eps_gx)
            cferror("error: (u, sigma, T) is not a solution such as sigma f^T(u) - u = 0");

        DedaField du(u);
        du.setToZero();
        // Set du  = EPS_du (random unit perturbation, "b" in Arnoldi A*b terms)
        if (duname.length() == 0) {
            cout << "Constructing du..." << endl;
            
            du = dsi->addPerturbations(1.0, decay);

            // bool meanflow_perturb = true;
            // du.addPerturbations(kxmax, kzmax, 1.0, decay, meanflow_perturb);

            // if (dnsflags.constraint == PressureGradient) {
            //     // Modify du so that (du/dy|a + du/dy|b) == (dw/dy|a + dw/dy|b) == 0
            //     // i.e. mean wall shear == 0
            //     Real h = (du.b() - du.a()) / 2;
            //     if (du.taskid() == du.task_coeff(0, 0)) {
            //         ChebyCoeff du00 = Re(du.profile(0, 0, 0));
            //         ChebyCoeff dw00 = Re(du.profile(0, 0, 2));
            //         ChebyCoeff du00y = diff(du00);
            //         ChebyCoeff dw00y = diff(dw00);
            //         Real duy = (du00y.eval_a() + du00y.eval_b()) / 2;
            //         Real dwy = (dw00y.eval_a() + dw00y.eval_b()) / 2;

            //         cout << "Modifying du so that it doesn't change mean pressure balance..." << endl;
            //         cout << "pre-mod : " << endl;
            //         cout << "(duya + duyb)/2 == " << duy << endl;
            //         cout << "(dwya + dwyb)/2 == " << dwy << endl;
            //         du.cmplx(0, 1, 0, 0) -=
            //             h * Complex(duy, 0);  // modify coeff of 1st chebyshev function T_1(y/h) = y/h for u
            //         du.cmplx(0, 1, 0, 2) -=
            //             h * Complex(dwy, 0);  // modify coeff of 1st chebyshev function T_1(y/h) = y/h for w

            //         du00 = Re(du.profile(0, 0, 0));
            //         dw00 = Re(du.profile(0, 0, 2));
            //         du00y = diff(du00);
            //         dw00y = diff(dw00);
            //         cout << "post-mod : " << endl;
            //         cout << "(duya + duyb)/2 == " << (du00y.eval_a() + du00y.eval_b()) / 2 << endl;
            //         cout << "(dwya + dwyb)/2 == " << (dw00y.eval_a() + dw00y.eval_b()) / 2 << endl;
            //     }
            // } else {  // (dnsflags.constraint == BulkVelocity)
            //     // modify du to have zero mean value
            //     if (du.taskid() == du.task_coeff(0, 0)) {
            //         cout << "Modifying du so that it doesn't change mean flow..." << endl;
            //         ChebyCoeff du00 = Re(du.profile(0, 0, 0));
            //         ChebyCoeff dw00 = Re(du.profile(0, 0, 2));
            //         Real umean = du00.mean();
            //         Real wmean = dw00.mean();
            //         cout << "pre-mod : " << endl;
            //         cout << "u mean == " << du00.mean() << endl;
            //         cout << "w mean == " << dw00.mean() << endl;
            //         du.cmplx(0, 0, 0, 0) -=
            //             Complex(umean, 0);  // modify coeff of 0th chebyshev function T_0(y/h) = 1 for u
            //         du.cmplx(0, 0, 0, 2) -=
            //             Complex(wmean, 0);  // modify coeff of 0th chebyshev function T_0(y/h) = 1 for w
            //         du00 = Re(du.profile(0, 0, 0));
            //         dw00 = Re(du.profile(0, 0, 2));
            //         cout << "post-mod : " << endl;
            //         cout << "u mean == " << du00.mean() << endl;
            //         cout << "w mean == " << dw00.mean() << endl;
            //     }
            // }
        } else {
            du.readNetCDF(duname);
        }

        // if (dnsflags.symmetries.length() != 0)
        //     project(dnsflags.symmetries, du);

        printout("L2Norm(du) = " + r2s(du.L2Norm()));
        printout("rescaling du by eps_du = " + r2s(EPS_du));
        du *= EPS_du / du.L2Norm();
        printout("L2Norm(du) = " + r2s(du.L2Norm()));

        VectorXd dx;
        du.toVector(dx);
        // dsi->makeVector(du, shifts_perturb, T, dx);

        E->solve(*dsi, x, dx, T, eps);

        // fftw_savewisdom();
    }
    MPI_Finalize();
}
