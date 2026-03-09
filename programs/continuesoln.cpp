/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#include <iostream>
#include "cfbasics/cfbasics.h"
#include "channelflow/dedadsi.h"
#include "channelflow/dedafield.h"
#include "nsolver/nsolver.h"

using namespace std;
using namespace Eigen;
using namespace chflow;

void readShifts(string shiftsname, std::map<string, Real>& shifts){

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

int main(int argc, char* argv[]) {
     MPI_Init(&argc, &argv);
    {
        ArgList args(argc, argv, "parametric continuation of invariant solution");

        ContinuationFlags cflags(args);
        cflags.save();

        unique_ptr<Newton> N;
        bool Rxsearch, Rzsearch, Tsearch;
        NewtonSearchFlags searchflags(args);
        searchflags.save();
        N = unique_ptr<Newton>(new NewtonAlgorithm(searchflags));

        Rxsearch = searchflags.xrelative;
        Rzsearch = searchflags.zrelative;
        Tsearch = searchflags.solntype == PeriodicOrbit ? true : false;
        //const bool Tnormalize = (Tsearch || searchflags.laurette) ? false : true;

        int W = 24;

        const Real Tinit = args.getreal("-T", "--T", "total time for integration");
        const string sys = args.getstr("-sys", "--system","active_matter", "system for time integration (python file name)");
        Real mui = args.getreal("-mui", "--MuInitial", -0.01, "Initial value of continuation parameter");

        args.section("Program options");
        const string muname = args.getstr(
            "-cont", "--continuation", "",
            "continuation parameter, one of [Re P Ub Uw ReP Theta ThLx ThLz Lx Lz Aspect Diag Lt Vs ReVs H HVs Rot]");
        const string shiftsname = args.getstr("-shifts", "--shifts",".", "filename for shifts in travelling wave");
        // const bool h5_read = args.getflag("-h5_read", "--read_h5", "read initial flowfield from h5 file");
        // const string h5_name = args.getstr("-h5_name","--hdf5_name",".","hdf5 filename");
        // const int h5_iter = args.getint("-h5_iter", "--hdf5_iter",0, "h5 read iteration");

        // check if invariant solution is relative

        bool relative = Rxsearch || Rzsearch || Tsearch;

        bool restart = cflags.restartMode;

        string uname(""), restartdir[3];
        if (restart) {
            bool solutionsAvail = readContinuationInfo(restartdir, cflags);

            if (!solutionsAvail) {
                restartdir[0] = args.getpath(1, "<string>", "directory containing solution 1");
                restartdir[1] = args.getpath(2, "<string>", "directory containing solution 2");
                restartdir[2] = args.getpath(3, "<string>", "directory containing solution 3");
            }

        } else {
            uname = args.getstr(1, "<flowfield>", "initial solution from which to start continuation");
        }
        args.check();

        if (muname == "") {
            cerr << "Please choose --continuation parameter"<< endl;
            exit(1);
        }
        args.save();

        DedaField u[3];
        std::map<string, Real> shifts[3];
        cfarray<Real> mu(3);
        Real T[3];
        unique_ptr<dedaDSI> dsi;
        // dsi = unique_ptr<dedaDSI>(new dedaDSI());

        if (restart) {
            cout << "Restarting from previous solutions. Please be aware that the DNSFlags "
                 << "from the corresponding directories will overwrite any specified command line parameters!" << endl;
            for (int i = 0; i < 3; ++i) {
                u[i].readNetCDF(restartdir[i] + "ubest");
            
                load(mu[i], restartdir[i] + "mu.asc");
                
                if (relative)
                    readShifts(restartdir[i] + "shiftsbest.asc", shifts[i]);

                T[i] = Tinit;
            }


        } else {  // not a restart
            // Compute initial data points for extrapolation from perturbations of given solution
            u[1].readNetCDF(uname);
            
            // if (h5_read)
            //     dsi->read(u[1], h5_name, h5_iter);
            u[2] = u[1];
            u[0] = u[1];

            T[0] = T[1] = T[2] = Tinit;

            if (relative){
                readShifts(shiftsname, shifts[0]);
                readShifts(shiftsname, shifts[1]);
                readShifts(shiftsname, shifts[2]);
            }
        }

    //     if (relative){
    //     ifstream is;
    //     string filename = ifstreamOpen(is, shiftsname, ".asc");
    //     if (!is) {
    //         cerr << "shifts : can't open file " << shiftsname << " or " << (shiftsname + ".asc")
    //             << endl;
    //         exit(1);
    //     }

    //     // Read in header. Form is "%N a b s"
    //     string comment;
    //     while (is.peek() == '%')
    //         getline(is, comment);
    //     int num_shifts;
    //     is >> num_shifts;
    //     for (int i = 0; i < num_shifts; i++){
    //         std::string dim;
    //         Real a;
    //         is>>dim>>a;
    //         shifts[dim] = a;
    //     }
    //     if (!is.good())
    //         cerr << "warning: bad istream in reading shifts from file " << filename << endl;
    //     is.close();
    // }

        cout << setw(4) << "i" << setw(W) << "T" << setw(W) << setw(W) << "L2Norm(u)" << endl;
        for (int i = 2; i >= 0; --i) {
            Real l2normui = u[i].L2Norm();
            cout << setw(4) << i << setw(W) << T[i] << setw(W) << l2normui << setw(W) << endl;
        }

        dsi = unique_ptr<dedaDSI>(new dedaDSI(Tinit, u[2], Tsearch, 
                            Rxsearch, Rzsearch, shifts[2], sys));

        cout << setprecision(8);
        printout("Working directory == " + pwd());
        printout("Command-line args == ");
        for (int i = 0; i < argc; ++i)
            cout << argv[i] << ' ';
        cout << endl;

        dsi->chooseMu(muname);
        dsi->updateMu(mui);
        if (!restart) {
            mu[1] = dsi->mu();
            mu[0] = mu[1] + cflags.initialParamStep;
            mu[2] = mu[1] - cflags.initialParamStep;
        }

        cfarray<VectorXd> x(3);
        for (int i = 0; i <= 2; ++i) {
            dsi->updateMu(mu[i]);
            dsi->makeVector(u[i], shifts[i], T[i], x[i]);
        }

        int Nunk = x[0].rows();
        int Nunk_total = Nunk;
#ifdef HAVE_MPI
        MPI_Allreduce(&Nunk, &Nunk_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
        cout << Nunk_total << " unknowns" << endl;

        Real muFinal = continuation(*dsi, *N, x, mu, cflags);
        cout << "Final mu is " << muFinal << endl;
    }
    MPI_Finalize();

    return 0;
}
