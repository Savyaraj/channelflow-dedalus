/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "cfbasics/mathdefs.h"
#include "channelflow/dedafield.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

// Decide if and how (if parallel) NetCDF may be used
#ifdef HAVE_NETCDF_PAR_H
#include <netcdf_par.h>
#define HAVE_NETCDF_PAR 1
#else
#define HAVE_NETCDF_PAR 0
#endif

#ifdef HAVE_NETCDF_H
#include <netcdf.h>
#define HAVE_NETCDF 1
#else
#define HAVE_NETCDF 0
#endif

#include <cstddef>  // for strtok
#include <cstdlib>
#include <cstring>  // for strdupa
#include <fstream>
#include <iomanip>
#include <vector>

using namespace std;
using namespace Eigen;

namespace chflow {

DedaField::DedaField() {}

DedaField::DedaField(int Nd, int Nvar, const int* N , const Real* L){
    resize(Nd, Nvar, N, L);
}

DedaField& DedaField::operator=(const DedaField& f) {
    resize(f.Nd_, f.Nvar_, f.N_.data(), f.L_.data());
    for (int i = 0; i < Nloc_; i++)
        rdata_[i] = f.rdata_[i];
    return *this;
}

DedaField& DedaField::operator-=(const DedaField& u) {
    for (int i = 0; i < Nloc_; i++)
        rdata_[i] -= u.rdata_[i];
    return *this;
}

DedaField& DedaField::operator*=(Real x){
    for (int i = 0; i < Nloc_; i++)
        rdata_[i] *= x;
    return *this;
}

DedaField& DedaField::operator/=(Real x){
    
    assert(x!=0);
    for (int i = 0; i < Nloc_; i++)
        rdata_[i] = rdata_[i]/x;
    return *this;
}

void DedaField::resize(int Nd, int Nvar, const int* N , const Real* L){
    
    assert(Nd >= 0);
    assert(Nvar >= 0);
    for (int j = 0 ; j < Nd ; j++){
        assert(N[j] >= 0);
        assert(L[j] >= 0);
    }

    Nd_ = Nd;
    Nvar_ = Nvar; 
    Nloc_ = Nvar;
    N_.resize(Nd);
    L_.resize(Nd);
    for (int j = 0 ; j < Nd ; j++){
        N_[j] = N[j];
        L_[j] = L[j];
        Nloc_ *= N[j];
    }

    rdata_.resize(Nloc_);

}

void DedaField::setToZero() {
    for (int i = 0; i < Nloc_; ++i)
        rdata_[i] = 0.0;
}

void DedaField::setRandom(int seed) {

    srand48(seed);
    for (int i = 0; i < Nloc_; ++i)
        rdata_[i] = rand();
}

Real DedaField::L2Norm() {

    Real sum = 0;
    for (int i = 0; i < Nloc_; i++)
        sum+= rdata_[i]*rdata_[i];
    sum = sum/Nloc_;
    Real norm = sqrt(sum);
    return norm;

}
void DedaField::save(const string& filebase, vector<string> component_names) const {
    string filename;

    /* suffix is given */
    if (HAVE_NETCDF)
        writeNetCDF(filebase, component_names);
    else {
        cferror(
            "DedaField::save(filename) error : can't save to NetCDF file because NetCDF libraries are not "
            "installed. filename == " +
            filebase);
    }
}

void DedaField::toVector(Eigen::VectorXd& v){

    if (v.size()<Nloc_)
        v.resize(Nloc_);

    for ( int i = 0; i < Nloc_; i++)
        v(i) = rdata_[i];

}

void DedaField::toField(const Eigen::VectorXd& v){
    
    for ( int i = 0; i < Nloc_; i++)
        rdata_[i] = v(i);

}

#define ERRCODE 2 /* Error message for the NetCDF methods */
#define ERR(e)                                \
    {                                         \
        cerr << "NetCDF: " << nc_strerror(e); \
        exit(ERRCODE);                        \
    }

#if HAVE_NETCDF
void DedaField::writeNetCDF(const string& filebase, vector<string> component_names) const {
    const int format_version = 1;

    // define and allocate local memory
    vector<int> N_io = N_;

    vector<Real> rdata_io = rdata_;

    /////////////////////////// NETCDF SETUP //////////////////////////////////////

    /* NetCDF return handles */
    int status;
    const int nd = Nd_;  // dimension of field (Nd_ is dimension of variable)
    int ncid;
    bool doesIO = false;

    auto dimid = vector<int>(nd, 0);
    auto gridid = vector<int>(nd, 0);
    auto varid = vector<int>(Nvar_, 0);

    /* Create the file, with parallel access if possible */
    string nc_name = appendSuffix(filebase, ".nc");

    if ((status = nc_create(nc_name.c_str(), NC_NETCDF4, &ncid)))
        ERR(status);
    doesIO = true;

    /* Define names for data description */
    vector<size_t> dim_size(nd);
    for (int i = 0; i < nd; i++){
        dim_size[i] = N_io[i];
    }

    vector<string> dim_name;
    for (int i = 0; i < nd; i++){
        dim_name.push_back("X_"+i2s(i));
    }
    vector<string> var_name;
    if (int(component_names.size()) == Nvar_) {
        for (int i = 0; i < Nvar_; i++){
            var_name.push_back(component_names[i]);
        }
    } 
    else {
    for (int i = 0; i < Nvar_; i++){
            var_name.push_back("Field_" + i2s(i));
    }

    if (doesIO) {  // header is handled either by master (without parNC) or by all tasks collectively (with parNC)
        /*define global attributes*/
        char cf_conv[] = "CF-1.0";
        char title[] = "DedaField";
        auto fversion = i2s(format_version);
        char reference[] = "Channelflow is free software: www.channelflow.ch.";
        time_t rawtime;  // current time
        struct tm* timeinfo;
        char tbuffer[80];
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(tbuffer, 80, "%Y-%m-%d %I:%M:%S", timeinfo);
        char hostname[1024];
        gethostname(hostname, 1023);
        /*write global attributes: CF (Climate and Forecast) convention*/
        if ((status = nc_put_att_text(ncid, NC_GLOBAL, "Conventions", strlen(cf_conv), cf_conv)))
            ERR(status);
        if ((status = nc_put_att_text(ncid, NC_GLOBAL, "title", strlen(title), title)))
            ERR(status);
        if ((status = nc_put_att_text(ncid, NC_GLOBAL, "format_version", fversion.size(), fversion.c_str())))
            ERR(status);
        if ((status = nc_put_att_text(ncid, NC_GLOBAL, "channelflow_version", strlen(CHANNELFLOW_VERSION),
                                        CHANNELFLOW_VERSION)))
            ERR(status);
        if ((status =
                    nc_put_att_text(ncid, NC_GLOBAL, "compiler_version", strlen(COMPILER_VERSION), COMPILER_VERSION)))
            ERR(status);
        if ((status = nc_put_att_text(ncid, NC_GLOBAL, "git_revision", strlen(g_GIT_SHA1), g_GIT_SHA1)))
            ERR(status);
        if ((status = nc_put_att_text(ncid, NC_GLOBAL, "time", strlen(tbuffer), tbuffer)))
            ERR(status);
        if ((status = nc_put_att_text(ncid, NC_GLOBAL, "host_name", strlen(hostname), hostname)))
            ERR(status);
        if ((status = nc_put_att_text(ncid, NC_GLOBAL, "references", strlen(reference), reference)))
            ERR(status);
        /*write global attributes: CF (ChannelFlow) requirements*/
        for (int i = 0; i < nd; i++){
            if ((status = nc_put_att_int(ncid, NC_GLOBAL, ("N_"+i2s(i)).c_str(), NC_INT, 1, &N_[i])))
            ERR(status);
        }
        for (int i = 0; i < nd; i++){
            if ((status = nc_put_att_double(ncid, NC_GLOBAL, ("L_"+i2s(i)).c_str(), NC_DOUBLE, 1, &L_[i])))
            ERR(status);
        }


        /* Define the dimensions and the grid variables (chosen here to have the same name) */
        for (int i = 0; i < nd; i++) {
            if ((status = nc_def_dim(ncid, dim_name[i].c_str(), dim_size[i], &dimid[nd - i - 1])))
                ERR(status);
            int dimid1[1];
            dimid1[0] = dimid[nd - i - 1];
            if ((status = nc_def_var(ncid, dim_name[i].c_str(), NC_DOUBLE, 1, dimid1, &gridid[i])))
                ERR(status);
        }

        /* Define the data variables. The type of the variable in this case is NC_DOUBLE (NC variable type = 6). */
        for (int i = 0; i < Nvar_; i++)
            if ((status = nc_def_var(ncid, var_name[i].c_str(), NC_DOUBLE, nd, dimid.data(), &varid[i])))
                ERR(status);

        /* End define mode. This tells netCDF we are done defining metadata. */
        if ((status = nc_enddef(ncid)))
            ERR(status);
    }

    /////////////////////////// NETCDF DATA OUTPUT //////////////////////////////////////

    /* define data size for parallel IO */
    int varsize = 1;
    for (int i = 0; i < Nd_; i++){
        varsize *= N_io[i];
    }
    auto var = vector<double>(varsize, 0.0);

    vector<size_t> start_var(nd), count_var(nd);
    if (varsize > 0) {
        for (int i = 0; i < Nd_; i++){
            start_var[i] = 0;
            count_var[i] = N_io[Nd_-1-i];
        }
    } else {  
        for (int i = 0; i < Nd_; i++){
            start_var[i] = 0;
            count_var[i] = 0;
        }
    }

    size_t start_grid[1], count_grid[1];

    if (doesIO) {
        for (int i = 0; i < Nd_; i++){
            auto grid = vector<double>(N_io[i], 0.0);
            for (int n = 0; n < N_io[i]; n++)
                grid[n] = n * L_[i] / N_io[i];
            start_grid[0] = 0;
            count_grid[0] = dim_size[i];
            if ((status = nc_put_vara_double(ncid, gridid[i], start_grid, count_grid, &grid[0])))
                ERR(status);
        }
        
    }

    /* Write the dedafield data to file */
    for (int i = 0; i < Nvar_; i++) {
        for (int n = 0; n < varsize; n++){
            var[n] = rdata_io[n + i*varsize];
        }
        /* Write NetCDF variable */
        if ((status = nc_put_vara_double(ncid, varid[i], start_var.data(), count_var.data(), var.data())))
            ERR(status);
    }

    }

    if (doesIO) {
        /* Close the file. This frees up any internal netCDF resources associated with the file, and flushes any
            * buffers. */
        if ((status = nc_close(ncid)))
            ERR(status);
    }

}

#endif

#if HAVE_NETCDF
void DedaField::readNetCDF(const string& filebase) {
    int status;  // return value
    int ncid;    // return handle of file

    /* Open the file, with parallel access if possible. */
    string nc_name = appendSuffix(filebase, ".nc");

    if ((status = nc_open(nc_name.c_str(), NC_NOWRITE, &ncid)))
        ERR(status);

    /* General inquiry of how many netCDF variables, dimensions, and global attributes are in the file. */
    auto ndims_in = 0;
    auto nvars_in = 0;
    auto ngatts_in = 0;
    auto unlimdimid_in = 0;

    vector<int> N_io;

    if ((status = nc_inq(ncid, &ndims_in, &nvars_in, &ngatts_in, &unlimdimid_in)))
        ERR(status);
    Nvar_ = nvars_in - ndims_in;  // the grid along each dimension is also counted as variables

    Nd_ = ndims_in; 
    N_io.resize(Nd_);
    N_.resize(Nd_);
    L_.resize(Nd_);

    /* Read necessary global attributes from file */
    char attname[NC_MAX_NAME + 1];
    for (int attid = 0; attid < ngatts_in; attid++) {  // loop over all global attributes
        if ((status = nc_inq_attname(ncid, NC_GLOBAL, attid, attname)))
            ERR(status);
        for (int i = 0; i< Nd_; i++){
            if (strcmp(attname, ("N_"+i2s(i)).c_str()) == 0) {
            if ((status = nc_get_att_int(ncid, NC_GLOBAL, attname, &N_[i])))
                ERR(status);
            } else if (strcmp(attname, ("L_"+i2s(i)).c_str()) == 0) {
            if ((status = nc_get_att_double(ncid, NC_GLOBAL, attname, &L_[i])))
                ERR(status);
            } else
            continue;
        }
    }

    /* Dimensions inquiry */
    char dimname[NC_MAX_NAME + 1];
    size_t dimlen;
    for (int dimid = 0; dimid < ndims_in; dimid++) {
        if ((status = nc_inq_dim(ncid, dimid, dimname, &dimlen)))
            ERR(status);
        /* Read the coordinate variable data. */
        for (int i = 0; i< Nd_; i++){
            if (strcmp(dimname, ("X_"+i2s(i)).c_str()) == 0){
                 N_io[i] = dimlen;
            }
        }
    }

    /* Define dedafield members and size of local hyperslab*/
    resize(Nd_, Nvar_, N_.data(), L_.data());

    /* Read in data variables from file */
    int ivar = 0;
    char varname[NC_MAX_NAME + 1];
    int varsize = 1;
    for (int i = 0; i < Nd_; i++){
        varsize *= N_io[i];
    }
    for (int varid = 0; varid < nvars_in; varid++) {
        if ((status = nc_inq_varname(ncid, varid, varname)))
            ERR(status);
        bool is_varname = 1;
        for (int i = 0; i < Nd_; i++){
            is_varname = is_varname && (strcmp(varname, ("X_"+i2s(i)).c_str()) != 0);
        }
        if (is_varname) {
            /* Define variable array and corners of the hyperslabs */
            auto var = vector<double>(varsize > 0 ? varsize : 1u, 0.0);

            auto start = vector<size_t>(ndims_in, 0);
            auto count = vector<size_t>(ndims_in, 0);

            if (varsize > 0) {
                for (int i = 0; i < Nd_; i++){
                    start[i] = 0;
                    count[i] = N_io[Nd_-1-i];
                }
            } else {  
                for (int i = 0; i < Nd_; i++){
                    start[i] = 0;
                    count[i] = 1;
                }
            }

            /* read data from file */

            if ((status = nc_get_vara_double(ncid, varid, start.data(), count.data(), var.data())))
                ERR(status);

            /* Construct DedaField */

            for (int n = 0; n < varsize; n++){
                rdata_[n + ivar*varsize] = var[n];
            }

            ivar++;
        } else
            continue;
    }


    /* Close the netcdf file. */
    if ((status = nc_close(ncid)))
        ERR(status);

}
#else
void DedaField::readNetCDF(const string& filebase) {
    cferror("FlowField::readNetCDF requires NetCDF libraries. Please install them and recompile channelflow.");
}
#endif  // HAVE_NETCDF

}  // namespace chflow
