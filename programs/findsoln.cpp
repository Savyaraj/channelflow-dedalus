/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include<channelflow/dedadsi.h>
#include "channelflow/dedafield.h"
#include "nsolver/nsolver.h"
#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/numpyconfig.h"

using namespace std;
using namespace Eigen;
using namespace chflow;

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    {
    ArgList args(argc, argv, "find an invariant solution using Newton-Krylov-hookstep algorithm");

    /** Choose the Newton algorithm to be used. Currently, two options are available: simple Newton without any
     * trust region optimization, and Newton with Hookstep (default). For the simple Newton, you can choose either a
     * full-space algorithm to solve the Newton equations (-solver "eigen") or between the two iterative algorithms
     * GMRES and BiCGStab. Newton-Hookstep requires GMRES. Note that the available parameters depend on your choice
     * of the algorithm.
     */

    unique_ptr<Newton> N;
    NewtonSearchFlags searchflags(args);
    searchflags.save(searchflags.outdir);
    N = unique_ptr<Newton>(new NewtonAlgorithm(searchflags));

    bool Rxsearch, Rzsearch, Tsearch;
    Rxsearch = searchflags.xrelative;
    Rzsearch = searchflags.zrelative;
    Tsearch = searchflags.solntype == PeriodicOrbit ? true : false;
    /** Read in remaining arguments */

    args.section("Program options");
    const string uname = args.getstr(1, "<flowfield>", "initial guess for the solution");
    const Real T = args.getreal("-T", "--T", "total time for integration");
    const string sys = args.getstr("-sys", "--system","active_matter", "system for time integration (python file name)");
    const bool mu_read = args.getflag("-mu_read", "--mu_read", "read parameter value");
    const string muname = args.getstr("-mu_name", "--mu_name",".", "parameter name");
    Real mu = args.getreal("-mu", "--Mu", 0.0, "Parameter value");
    const string shiftsname = args.getstr("-shifts", "--shifts",".", "filename for shifts in travelling wave");
    const bool h5_read = args.getflag("-h5_read", "--read_h5", "read initial flowfield from h5 file");
    const string h5_name = args.getstr("-h5_name","--hdf5_name",".","hdf5 filename");
    const int h5_iter = args.getint("-h5_iter", "--hdf5_iter",0, "h5 read iteration");


    args.check();
    args.save();
    DedaField u;
    u.readNetCDF(uname);
    std::map<string, Real> shifts;
    if (Rxsearch||Rzsearch||Tsearch){
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
        }
        if (!is.good())
            cerr << "warning: bad istream in reading shifts from file " << filename << endl;
        is.close();
    }
    /** Construct the dynamical-systems interface object depending on the given parameters. Current options are
     * either standard (f(u) via forward time integration) or Laurette (f(u) via Laurettes method)
     */

    unique_ptr<dedaDSI> dsi;
    dsi = unique_ptr<dedaDSI>(new dedaDSI(T, u, Tsearch, 
                            Rxsearch, Rzsearch, shifts, sys));

    if (mu_read){
        dsi->chooseMu(muname);
        dsi->updateMu(mu);
    }
    if (h5_read)
        dsi->read(u, h5_name, h5_iter);

    VectorXd x_singleShot;
    VectorXd x;
    VectorXd yvec;
    MatrixXd y;
    MultishootingDSI* msDSI = N->getMultishootingDSI();
    dsi->makeVector(u, shifts, T, x_singleShot);
    msDSI->setDSI(*dsi, x_singleShot.size());
    x = msDSI->makeMSVector(x_singleShot);

    int Nunk = x.size();
    int Nunk_total = Nunk;
#ifdef HAVE_MPI
    MPI_Allreduce(&Nunk, &Nunk_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    cout << Nunk_total << " unknowns" << endl;

    Real residual = 0;
    VectorXd x_out;
    x_out = N->solve(*dsi, x, residual);
    }
    MPI_Finalize();
    return 0;
}
