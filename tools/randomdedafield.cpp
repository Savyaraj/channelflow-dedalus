/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <fstream>
#include <iomanip>
#include <iostream>
#include "channelflow/dedafield.h"
#include "channelflow/utilfuncs.h"

using namespace std;
using namespace chflow;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    {
        string purpose(
            "Construct a random dedafield with rescaling\n"
            "u = magnitude/L2Norm(u)");

        ArgList args(argc, argv, purpose);

        const int Nd = args.getint("-Nd", "--Nd", "# dimensions");
        const int Nvar = args.getint("-Nvar", "--Nvar", "# variables");
        const string N_str = args.getstr("-N", "--N", "", "# gridpoints along each dimensions, seperated by comma");
        const string L_str = args.getstr("-L", "--L", "", "gridsize along each dimensions, seperated by comma");
        const int seed = args.getint("-sd", "--seed", 1, "seed for random number generator");
        const Real magn = args.getreal("-m", "--magnitude", 0.20, "magnitude  of field, 0 < m < 1");
        const string uname = args.getstr(1, "<fieldname>", "output file");
        
        args.check();
        args.save("./");

        vector<int> N;
        vector<Real>L;

        stringstream N_str_stream(N_str);
        while( N_str_stream.good() )
        {
            string substr;
            getline( N_str_stream, substr, ',' );
            N.push_back( stoi(substr.c_str()) );
        }

        stringstream L_str_stream(L_str);
        while( L_str_stream.good() )
        {
            string substr;
            getline( L_str_stream, substr, ',' );
            L.push_back( stod(substr.c_str()) );
        }

        DedaField u(Nd, Nvar, N.data(), L.data());
        u.setRandom(seed);
        u *= magn / u.L2Norm();
        u.save(uname);
    }
    MPI_Finalize();
}
