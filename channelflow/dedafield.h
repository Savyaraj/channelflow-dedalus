/**
 * Class for inteface between dedalus fields and nsolver 
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef DEDAFIELD_H
#define DEDAFIELD_H

#include "cfbasics/cfvector.h"
#include "cfbasics/mathdefs.h"
#include <Eigen/Dense>

#include <fftw3.h>
#include <memory>
#include <vector>
using namespace std;
using namespace Eigen;

namespace chflow {

class DedaField {
   public:
   // Constructors
    DedaField();
    DedaField(int Nd, int Nvar, const int* N , const Real* L);

    // Operators
    DedaField& operator=(const DedaField& u);  // assign identical copy of U
    DedaField& operator-=(const DedaField& u);
    DedaField& operator*=(Real x);
    DedaField& operator/=(Real x);
    void resize(int Nd, int Nvar, const int* N , const Real* L);
    void setToZero();
    void setRandom(int seed);
    Real L2Norm();

    inline std::vector<int> N() const;  // number of gridpoints
    inline std::vector<Real> L() const; // domain size
    inline int Nd() const;              // number of dimensions
    inline int Nvar() const;            // number of variables
    inline lint Nloc() const;            // length of flattened array
    
    // i/o methods
    void writeNetCDF(const std::string& filebase,
                     std::vector<std::string> component_names = std::vector<std::string>()) const;
    void readNetCDF(const std::string& filebase);
    void save(const std::string& filebase, std::vector<std::string> component_names = std::vector<std::string>()) const;
    
    // conversion from fields to Eigen vectors
    void toVector(VectorXd& v);
    void toField(const VectorXd& v);
    inline std::vector<Real> get_rdata() const; 
    inline void set_rdata(const Real* v);

   private:
    int Nd_ = 0;           // number of dimensions
    int Nvar_ = 0;         // number of variables
    std::vector<int> N_;   // number of gridpoints along each dimension
    std::vector<Real> L_;  // domain size along each dimension
    lint Nloc_;            // total number of gridpoints

    // Storage in flattened array
    std::vector<Real> rdata_;
};

inline std::vector<int> DedaField::N() const { return N_; }
inline int DedaField::Nd() const { return Nd_; }
inline int DedaField::Nvar() const { return Nvar_; }
inline lint DedaField::Nloc() const { return Nloc_; }
inline std::vector<Real> DedaField::L() const { return L_; }
inline std::vector<Real> DedaField::get_rdata() const {
    return rdata_;
}
inline void DedaField::set_rdata(const Real* v){
    for (int i = 0 ; i < Nloc_ ; i++ )
        rdata_[i] = v[i];
}

}  //namespace chflow
#endif
