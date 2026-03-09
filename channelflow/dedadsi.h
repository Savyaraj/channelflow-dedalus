/**
 * Dynamical System Interface (DSI) for communicating with dedalus 
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */// #ifdef NPY_1_7_API_VERSION
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifndef DEDADSI_H
#define DEDADSI_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NPE_PY_ARRAY_OBJECT PyArrayObject

#include <sys/stat.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include "cfbasics/cfvector.h"
#include "channelflow/dedafield.h"

#include "nsolver/nsolver.h"
#include <Eigen/Dense>

#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/numpyconfig.h"


namespace chflow {

// converts the string from "fieldstats" in diffops to a vector of Reals
// std::vector<Real> fieldstats_vector(const FlowField& u);

class dedaDSI : public DSI {
   public:
   
    /** \brief default constructor */
    dedaDSI();
    virtual ~dedaDSI() {}

    /** \brief Initialize dedalusDSI */
    dedaDSI(Real T, const DedaField& u, bool Tsearch, bool xrelative, bool zrelative, std::map<string, Real> shifts, std::string sys, std::ostream* os = &std::cout);
    Eigen::VectorXd eval(const Eigen::VectorXd& x) override;
    Eigen::VectorXd eval(const Eigen::VectorXd& x0, const Eigen::VectorXd& x1, bool symopt) override;
    void save(const Eigen::VectorXd& x, const std::string filebase, const std::string outdir = "./",
              const bool fieldsonly = false) override;
    void saveShifts(const string filebase, const string outdir, std::map<string, Real> shifts, Real T, std::ios::openmode openflag = std::ios::out);
    void read(DedaField& u, std::string filebase, int iter);
    void saveEigenvec(const Eigen::VectorXd& x, const std::string label, const std::string outdir) override;
    void saveEigenvec(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const std::string label1,
                      const std::string label2, const std::string outdir) override;
    void makeVector(DedaField& u, std::map<string, Real>& shifts, const Real T, Eigen::VectorXd& x);
    void extractVector(const Eigen::VectorXd& x, DedaField& u, std::map<string, Real>& shifts, Real& T);
    
    Eigen::VectorXd xdiff(const Eigen::VectorXd& a) override;
    Eigen::VectorXd zdiff(const Eigen::VectorXd& a) override;
    Eigen::VectorXd tdiff(const Eigen::VectorXd& a, Real epsDt) override;
    DedaField addPerturbations(Real mag, Real decay);
    Real extractT(const Eigen::VectorXd& x) override;
    Real extractXshift(const Eigen::VectorXd& x) override;
    Real extractZshift(const Eigen::VectorXd& x) override;
    Real observable(VectorXd& x);

    Real tph_observable(Eigen::VectorXd& x) override;
    bool XrelSearch() const override { return xrelative_; };
    bool ZrelSearch() const override { return zrelative_; };
    bool Tsearch() const override { return Tsearch_; };
    
    /// \name Handle continuation parameter
    void updateMu(Real mu);
    void chooseMu(std::string muName);
    std::string printMu();  // document
    void saveParameters(std::string searchdir);
    string statsHeader();
    string stats(const VectorXd& x);
    // void save_array(DedaField& u);
    // void advance(DedaField& u, DedaField& f_u, Real T);

   protected:
    PyObject* de_;
    Real Tinit_;
    std::vector<int> N_;
    int Nd_ = 0;
    int Nvar_ = 0;
    std::vector<Real> L_;
    bool Tsearch_;
    bool xrelative_;
    bool zrelative_;
    std::map<string, Real> shifts_;
    std::string sys_;

   private:
    
    std::string muName_;
    Real mu_;
};


PyObject* initDedalus(std::string sys);

void advanceDedalus(DedaField& u, Real T, PyObject* de, bool xrelative, bool zrelative, std::map<string, Real>& shifts); 

void f(const DedaField& u, Real T, DedaField& f_u, PyObject* de, bool xrelative, bool zrelative, std::map<string, Real>& shifts, std::ostream& os);

void G(const DedaField& u, Real& T, DedaField& Gu, PyObject*de, bool xrelative, bool zrelative, std::map<string, Real>& shifts, std::ostream& os = std::cout);

}

#endif