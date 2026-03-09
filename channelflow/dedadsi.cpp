/**
 * Channelflow Dynamical System Interface
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "dedadsi.h"
#include"channelflow/dedafield.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPE_PY_ARRAY_OBJECT PyArrayObject

#include <Python.h>
#include "numpy/arrayobject.h"
#include <Eigen/Dense>
#include "numpy/numpyconfig.h"

using namespace std;
using namespace Eigen;

namespace chflow {

dedaDSI::dedaDSI() {}

dedaDSI::dedaDSI(Real T,const DedaField& u, bool Tsearch, bool xrelative, bool zrelative, std::map<string, Real> shifts, std::string sys, ostream* os)
        :DSI(os),
        de_(initDedalus(sys)),
        Tinit_(T),
        N_(u.N()),
        Nd_(u.Nd()),
        Nvar_(u.Nvar()),
        L_(u.L()),
        Tsearch_(Tsearch),
        xrelative_(xrelative),
        zrelative_(zrelative),
        shifts_(shifts),
        sys_(sys),
        muName_("None"),
        mu_(0){}

VectorXd dedaDSI::eval(const VectorXd& x) {
    DedaField u(Nd_, Nvar_, N_.data(), L_.data());
    Real T;
    extractVector(x, u, shifts_, T);
    DedaField Gu(Nd_, Nvar_, N_.data(), L_.data());
    G(u, T, Gu, de_, xrelative_, zrelative_, shifts_, *os_);
    VectorXd Gx(VectorXd::Zero(x.size()));
    Gu.toVector(Gx);
    return Gx;
}

VectorXd dedaDSI::eval(const VectorXd& x0, const VectorXd& x1, bool symopt) {
    DedaField u0(Nd_, Nvar_, N_.data(), L_.data());
    DedaField u1(Nd_, Nvar_, N_.data(), L_.data());
    Real T0, T1;
    std::map<string, Real> shifts;
    extractVector(x0, u0, shifts, T0);
    extractVector(x1, u1, shifts, T1);
    DedaField Gu(Nd_, Nvar_, N_.data(), L_.data());

    f(u0, T0, Gu, de_, xrelative_, zrelative_, shifts, *os_);
    Gu*= -T0;
    Gu -= u0;
    Gu*= -1;
    Gu -= u1;
    
    VectorXd Gx(VectorXd::Zero(x0.size()));
    Gu.toVector(Gx);  
    return Gx;

}


//----------------------TODO----------------------------
void dedaDSI::save(const VectorXd& x, const string filebase, const string outdir, const bool fieldsonly) {
    DedaField u(Nd_, Nvar_, N_.data(), L_.data());
    Real T;
    std::map<string, Real> shifts;
    extractVector(x, u, shifts, T);
    u.save(outdir + "u" + filebase);
    if (!fieldsonly){
        VectorXd x;
        u.toVector(x);
        string fs = stats(x);
        ofstream fout((outdir + "fieldconverge.asc").c_str(), ios::app);
        long pos = fout.tellp();
        if (pos == 0)
            fout << statsHeader() << endl;
        fout << fs << endl;
        fout.close();
        if (xrelative_ || zrelative_ || Tsearch_)
            saveShifts(filebase, outdir, shifts, T);
    }

}

void dedaDSI::saveShifts(const string filebase, const string outdir, std::map<string, Real> shifts, Real T, std::ios::openmode openflag){
    
    ofstream out((outdir + "shifts"+filebase+".asc").c_str(), openflag);
    out << setprecision(17);
    int Nshifts = xrelative_ + zrelative_ + Tsearch_;
    out << Nshifts << endl;
    if (Tsearch_)
        out << "T" << "\t" << shifts["T"] << endl;
    if (xrelative_)
        out << "x" << "\t" << shifts["x"] << endl;
    if (zrelative_)
        out << "z" << "\t" << shifts["z"] << endl;
    out.close();

}

void dedaDSI::read(DedaField& u, std::string filebase, int iter){

    PyObject* name_py = PyUnicode_FromString(filebase.c_str());
    PyObject* iter_py = Py_BuildValue("i", iter);
	PyObject *method =  PyUnicode_FromString("read_h5");
	PyObject *u_out = PyObject_CallMethodObjArgs(de_, method, name_py, iter_py, NULL);

    if (u_out == NULL) {
        PyErr_Print();
        Py_DECREF(name_py);
        Py_DECREF(iter_py);
        Py_DECREF(method);
        cferror("Failed to call Python read_h5 method");
    }

    npy_intp u_c_ind[1]{0};
	Real* u_c = reinterpret_cast<Real*>(PyArray_GetPtr((PyArrayObject*)u_out, u_c_ind));

    if (u_c == NULL) {
        PyErr_Print();
        Py_DECREF(name_py);
        Py_DECREF(iter_py);
        Py_DECREF(method);
        Py_DECREF(u_out);
        cferror("Failed to extract array data from Python");
    }

	//Convert this array back to Flowfield format
    u.set_rdata(u_c);
    Py_DECREF(name_py);
    Py_DECREF(iter_py);
    Py_DECREF(method);
    Py_DECREF(u_out);
}

void dedaDSI::saveEigenvec(const VectorXd& ev, const string label, const string outdir) {
    DedaField ef(Nd_, Nvar_, N_.data(), L_.data());
    ef.toField(ev);
    ef.save(outdir + "ef" + label);
}

void dedaDSI::saveEigenvec(const VectorXd& evA, const VectorXd& evB, const string label1, const string label2,
                         const string outdir) {
    DedaField efA(Nd_, Nvar_, N_.data(), L_.data());
    DedaField efB(Nd_, Nvar_, N_.data(), L_.data());
    efA.toField(evA);
    efB.toField(evB);
    // Real c = 1.0 / sqrt(L2Norm2(efA) + L2Norm2(efB));
    // efA *= c;
    // efB *= c;
    efA.save(outdir + "ef" + label1);
    efB.save(outdir + "ef" + label2);
}

void dedaDSI::makeVector(DedaField& u, std::map<string, Real>& shifts, const Real T, Eigen::VectorXd& x){
    int uunk = u.Nloc();                         // # of variables for u unknonwn
    const int Tunk = Tsearch_ ? uunk : -1;  // index for T unknown
    const int xunk = xrelative_ ? uunk + Tsearch_ : -1;
    const int zunk = zrelative_ ? uunk + Tsearch_ + xrelative_ : -1;
    int Nunk = uunk + Tsearch_ + xrelative_ + zrelative_;

    Eigen::VectorXd v;
    u.toVector(v);
    if (x.rows() < Nunk)
        x.resize(Nunk);
    int ind = 0;
    for ( int i = 0; i < u.Nloc(); i++){
        x[ind] = v[i];
        ind++;
    }

    if (Tsearch_)
        x(Tunk) = shifts["T"];
    if (xrelative_)
        x(xunk) = shifts["x"];
    if (zrelative_)
        x(zunk) = shifts["z"];

}

void dedaDSI::extractVector(const Eigen::VectorXd& x, DedaField& u, std::map<string, Real>& shifts, Real& T){
    u.resize(Nd_, Nvar_, N_.data(), L_.data());
    Eigen::VectorXd v(u.Nloc());
    int ind = 0;
    for ( int i = 0; i < u.Nloc(); i++){
        v[ind] = x[i];
        ind++;
    }
    u.toField(v);
    int uunk = u.Nloc();  
    const int Tunk = uunk + Tsearch_ - 1;
    const int xunk = uunk + Tsearch_ + xrelative_ - 1;
    const int zunk = uunk + Tsearch_ + xrelative_ + zrelative_ - 1;

    T = Tsearch_ ? x(Tunk) : Tinit_;
    if (Tsearch_)
        shifts["T"] = x(Tunk);
    if (xrelative_)
        shifts["x"] = x(xunk);
    if (zrelative_)
        shifts["z"] = x(zunk);
    
}

VectorXd dedaDSI::xdiff(const VectorXd& a) {
    DedaField u(Nd_, Nvar_, N_.data(), L_.data());
    u.toField(a);
    VectorXd dadx(a.size());

    vector<Real> rdata = u.get_rdata();
    vector<double> double_array(rdata.begin(),rdata.end());
	double* Array = double_array.data();
    int double_array_size = double_array.size();
    PyObject *py_array;
	double *ptr = Array;
	npy_intp dims[1] = { double_array_size };
	py_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, ptr);
	PyObject *method =  PyUnicode_FromString("diff");
    PyObject *dir =  PyUnicode_FromString("x");
	PyObject *u_out = PyObject_CallMethodObjArgs(de_, method, py_array, dir, NULL);

    if (u_out == NULL) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_DECREF(method);
        Py_DECREF(dir);
        cferror("Failed to call Python diff method for x direction");
    }

    npy_intp u_c_ind[1]{0};
	Real* u_c = reinterpret_cast<Real*>(PyArray_GetPtr((PyArrayObject*)u_out, u_c_ind));

    if (u_c == NULL) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_DECREF(method);
        Py_DECREF(dir);
        Py_DECREF(u_out);
        cferror("Failed to extract array data from Python diff");
    }

    u.set_rdata(u_c);
    u.toVector(dadx);
    dadx *= 1. / dadx.norm();

    Py_DECREF(py_array);
    Py_DECREF(method);
    Py_DECREF(dir);
    Py_DECREF(u_out);
    return dadx;

}

VectorXd dedaDSI::zdiff(const VectorXd& a) {
    DedaField u(Nd_, Nvar_, N_.data(), L_.data());
    u.toField(a);
    VectorXd dadz(a.size());

    vector<Real> rdata = u.get_rdata();
    vector<double> double_array(rdata.begin(),rdata.end());
	double* Array = double_array.data();
    int double_array_size = double_array.size();
    PyObject *py_array;
	double *ptr = Array;
	npy_intp dims[1] = { double_array_size };
	py_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, ptr);
	PyObject *method =  PyUnicode_FromString("diff");
    PyObject *dir =  PyUnicode_FromString("y");
	PyObject *u_out = PyObject_CallMethodObjArgs(de_, method, py_array, dir, NULL);

    if (u_out == NULL) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_DECREF(method);
        Py_DECREF(dir);
        cferror("Failed to call Python diff method for y direction");
    }

    npy_intp u_c_ind[1]{0};
	Real* u_c = reinterpret_cast<Real*>(PyArray_GetPtr((PyArrayObject*)u_out, u_c_ind));

    if (u_c == NULL) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_DECREF(method);
        Py_DECREF(dir);
        Py_DECREF(u_out);
        cferror("Failed to extract array data from Python diff");
    }

    u.set_rdata(u_c);

    u.toVector(dadz);
    dadz *= 1. / dadz.norm();

    Py_DECREF(py_array);
    Py_DECREF(method);
    Py_DECREF(dir);
    Py_DECREF(u_out);
    return dadz;

}

VectorXd dedaDSI::tdiff(const VectorXd& a, Real epsDt) {
    DedaField u(Nd_, Nvar_, N_.data(), L_.data());
    u.toField(a);
    DedaField edudtf(Nd_, Nvar_, N_.data(), L_.data());
    f(u, epsDt, edudtf, de_, xrelative_, zrelative_, shifts_, *os_);
    //edudtf -= u;
    VectorXd dadt(a.size());
    edudtf.toVector(dadt);
    if (dadt.norm() > 0)
        dadt *= 1. / dadt.norm();
    return dadt;
}

DedaField dedaDSI::addPerturbations(Real mag, Real decay){

    DedaField du(Nd_, Nvar_, N_.data(), L_.data());

    PyObject *method =  PyUnicode_FromString("add_perturbations");
    PyObject* mag_py = Py_BuildValue("d", mag);
    PyObject* decay_py = Py_BuildValue("d", decay);
	PyObject *u_out = PyObject_CallMethodObjArgs(de_, method, mag_py, decay_py, NULL);

    if (u_out == NULL) {
        PyErr_Print();
        Py_DECREF(method);
        Py_DECREF(mag_py);
        Py_DECREF(decay_py);
        cferror("Failed to call Python add_perturbations method");
    }

    npy_intp u_c_ind[1]{0};
	Real* u_c = reinterpret_cast<Real*>(PyArray_GetPtr((PyArrayObject*)u_out, u_c_ind));

    if (u_c == NULL) {
        PyErr_Print();
        Py_DECREF(method);
        Py_DECREF(mag_py);
        Py_DECREF(decay_py);
        Py_DECREF(u_out);
        cferror("Failed to extract array data from Python add_perturbations");
    }

    du.set_rdata(u_c);

    Py_DECREF(method);
    Py_DECREF(mag_py);
    Py_DECREF(decay_py);
    Py_DECREF(u_out);
    return du;

}

Real dedaDSI::extractT(const VectorXd& x) {
    Real Tvec;
    std::map<string, Real> shifts;
    DedaField u(Nd_, Nvar_, N_.data(), L_.data());
    extractVector(x, u, shifts, Tvec);
    return Tvec;
}

Real dedaDSI::extractXshift(const VectorXd& x) {
    Real Tvec;
    std::map<string, Real> shifts;
    DedaField u(Nd_, Nvar_, N_.data(), L_.data());
    extractVector(x, u, shifts, Tvec);
    return shifts["x"];
}

Real dedaDSI::extractZshift(const VectorXd& x) {
    Real Tvec;
    std::map<string, Real> shifts;
    DedaField u(Nd_, Nvar_, N_.data(), L_.data());
    extractVector(x, u, shifts, Tvec);
    return shifts["z"];
}

Real dedaDSI::observable(VectorXd& x) {

    DedaField u(Nd_, Nvar_, N_.data(), L_.data());
    u.toField(x);

    vector<Real> rdata = u.get_rdata();
    vector<double> double_array(rdata.begin(),rdata.end());
	double* Array = double_array.data();
    int double_array_size = double_array.size();
    PyObject *py_array;
	double *ptr = Array;
	npy_intp dims[1] = { double_array_size };
	py_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, ptr);
	PyObject *method =  PyUnicode_FromString("observable");
	PyObject *obs_py = PyObject_CallMethodObjArgs(de_, method, py_array, NULL);

    if (obs_py == NULL) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_DECREF(method);
        cferror("Failed to call Python observable method");
    }

    Real obs = PyFloat_AsDouble(obs_py);

    Py_DECREF(py_array);
    Py_DECREF(method);
    Py_DECREF(obs_py);
    return obs;

}

Real dedaDSI::tph_observable(VectorXd& x) {
    
    return observable(x);

}

void dedaDSI::updateMu(Real mu) {
    
    DSI::updateMu(mu);
    mu_ = mu;    
    PyObject *method =  PyUnicode_FromString("updateMu");
    PyObject* mu_py = Py_BuildValue("d", mu);
    PyObject* muName_py = Py_BuildValue("s", muName_.c_str());

	PyObject *result = PyObject_CallMethodObjArgs(de_, method, mu_py, muName_py, NULL);

    if (result == NULL) {
        PyErr_Print();
        Py_DECREF(method);
        Py_DECREF(mu_py);
        Py_DECREF(muName_py);
        cferror("Failed to call Python updateMu method");
    }

    Py_DECREF(method);
    Py_DECREF(mu_py);
    Py_DECREF(muName_py);
    Py_DECREF(result);
}

void dedaDSI::chooseMu(string muName) { muName_ = muName; }

string dedaDSI::printMu() { return muName_; }

//-------------------TODO-----------------------------------
void dedaDSI::saveParameters(string searchdir) { return; }

string dedaDSI::statsHeader(){

    stringstream header;
    header << setw(8) << "#(" << muName_ << ")" << setw(14) << "L2";
    if (Tsearch_)
        header << setw(14) << "L2_max" << setw(14) << "L2_min";
    return header.str();

}

string dedaDSI::stats(const VectorXd& x){

    DedaField u(Nd_, Nvar_, N_.data(), L_.data());
    Real T;
    std::map<string, Real> shifts;
    extractVector(x, u, shifts, T);
    stringstream s;
    s << setw(8);
    s << mu_;
    Eigen::VectorXd x_temp;
    u.toVector(x_temp);
    Real norm = observable(x_temp);
    s << setw(14) << norm;
    Real norm_max = norm;
    Real norm_min = norm;
    if (Tsearch_ && T > 1.0e-7){
        Real timep = T / 100.0;
        DedaField Gu(u); 
        for (int t = 0; t < 100; t++){
            f(u, timep, Gu, de_, xrelative_, zrelative_, shifts, *os_);
            Gu*= -timep;
            Gu -= u;
            Gu*= -1;
            u = Gu;
            u.toVector(x_temp);
            Real new_norm = observable(x_temp);
            norm_max = (norm_max > new_norm) ? norm_max : new_norm;
            norm_min = (norm_min < new_norm) ? norm_min : new_norm;
        }
    }
    s << setw(14) << norm_max << setw(14) << norm_min;
    return s.str();
    
}

PyObject* initDedalus(std::string sys){

	//Building python module and creating DedalusPy object
	PyObject *module_name, *module, *dict, *python_class, *object;

	Py_Initialize();
    import_array();
    PyRun_SimpleString("from dedalus import public as de");
	module_name = PyUnicode_FromString(sys.c_str()); //active_matter
	module = PyImport_Import(module_name);
    if (module == NULL){
        PyErr_Print();
        Py_DECREF(module_name);
        cferror("Failed to import Python module: " + sys);
    }
	Py_DECREF(module_name);
	dict = PyModule_GetDict(module);
	Py_DECREF(module);
	python_class = PyDict_GetItemString(dict, "DedalusPy");
	Py_DECREF(dict);
	object = PyObject_CallObject(python_class, nullptr);
	Py_DECREF(python_class); 
    return object;    
}

void advanceDedalus(DedaField& u, Real T, PyObject* de, bool xrelative, bool zrelative, std::map<string, Real>& shifts){

	// Create an array to pass in dedalus
    vector<Real> rdata = u.get_rdata();
    int uunk = u.Nloc();  
    const int xunk = uunk + xrelative - 1;
    const int zunk = uunk + xrelative + zrelative - 1;
    int Nunk = zunk + 1;
    vector<double> Array(Nunk);
    for (int i = 0; i < uunk; i++)
        Array[i] = rdata[i];

    if (xrelative)
        Array[xunk] = shifts["x"];
    if (zrelative)
        Array[zunk] = shifts["z"];
    
    int double_array_size = Nunk;
    PyObject *py_array;
	double *ptr = &Array[0];
	npy_intp dims[1] = { double_array_size };
	py_array = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, ptr);
    PyObject* T_py = Py_BuildValue("d", T);
	PyObject *method =  PyUnicode_FromString("advance");
	PyObject *u_out = PyObject_CallMethodObjArgs(de, method, T_py, py_array, NULL);

    if (u_out == NULL) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_DECREF(T_py);
        Py_DECREF(method);
        cferror("Failed to call Python advance method");
    }

    npy_intp u_c_ind[1]{0};
	Real* u_c = reinterpret_cast<Real*>(PyArray_GetPtr((PyArrayObject*)u_out, u_c_ind));

    if (u_c == NULL) {
        PyErr_Print();
        Py_DECREF(py_array);
        Py_DECREF(T_py);
        Py_DECREF(method);
        Py_DECREF(u_out);
        cferror("Failed to extract array data from Python advance");
    }

    if (xrelative)
        shifts["x"] = u_c[xunk];
    if (zrelative)
        shifts["z"] = u_c[zunk];

	//Convert this array back to Flowfield format
    u.set_rdata(u_c);

    Py_DECREF(py_array);
    Py_DECREF(T_py);
    Py_DECREF(method);
    Py_DECREF(u_out);
}

// return f^{N dt}(u) = time-(N dt) DNS integration of u
void f(const DedaField& u, Real T, DedaField& f_u, PyObject* de, bool xrelative, bool zrelative, std::map<string, Real>& shifts, ostream& os){
    DedaField u_temp = u;
    advanceDedalus(u_temp, T, de, xrelative, zrelative, shifts);
    f_u = u_temp;
    return;
}

// G(x) = G(u,sigma) = (sigma f^T(u) - u) for orbits
void G(const DedaField& u, Real& T, DedaField& Gu, PyObject* de,
       bool xrelative, bool zrelative, std::map<string, Real>& shifts, ostream& os){
    f(u, T, Gu, de, xrelative, zrelative, shifts, os);
}


}  // namespace chflow