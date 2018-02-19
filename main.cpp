#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <sys/time.h>
#include <stddef.h>
#include <netcdf.h>
#include <boost/numeric/odeint.hpp>
#include <boost/random.hpp>
#include <boost/array.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random/normal_distribution.hpp>
#include "mpi.h"

using namespace std;
using namespace boost::numeric::odeint;
typedef double value_type;
typedef std::vector<value_type> state_type;
typedef boost::mt19937 RNGType;         // mersenne twister generator
MPI_Datatype mpi_type = MPI_DOUBLE;
nc_type NC_TYPE = NC_DOUBLE;
#include "class.h"

#define R 1.987     // Universal gas constant [cal/mol]


template< size_t N > class SDE_Euler
{
    public:
    typedef boost::array< double , N > state_type;
    typedef boost::array< double , N > deriv_type;
    typedef unsigned short order_type;
    typedef boost::numeric::odeint::stepper_tag stepper_category;

    static order_type order( void ) { return 1; }

    template< class System >
    void do_step( System system , state_type &f , value_type t , value_type dt ) const
    {
        deriv_type det, stoch ;
        system.first(f, det);
        system.second(f, stoch);
        for (size_t i=0; i<N; ++i) {
            f[i] += dt * det[i] + sqrt(dt) * stoch[i];        
        }
    }
};


template<size_t N> struct VectorDeterministic
{
    ParameterClassVector m_para;
    NeighborsClass m_neigh;
    value_type m_dt;

    public:
    VectorDeterministic( ParameterClassVector para, NeighborsClass neighbors ) 
    : m_para(para), m_neigh(neighbors) {}
    typedef boost::array<value_type, N> state_type;

    void operator() ( const state_type &x , state_type &f ) const
    {
        value_type b = m_para.b;
        value_type rho_E = m_para.rho_E;
        value_type rho_L = m_para.rho_L;
        value_type rho_P = m_para.rho_P;
        value_type rho_Ah = m_para.rho_Ah;
        value_type rho_Ar = m_para.rho_Ar;
        value_type rho_Ao = m_para.rho_Ao;
        value_type mu_E = m_para.mu_E;
        value_type mu_L1 = m_para.mu_L1;
        value_type mu_L2 = m_para.mu_L2;
        value_type mu_P = m_para.mu_P;
        value_type mu_Ah = m_para.mu_Ah;
        value_type mu_Ar = m_para.mu_Ar;
        value_type mu_Ao = m_para.mu_Ao;
        value_type psi_H = m_para.psi_H;
        value_type psi_B = m_para.psi_B;
        NeighborsClass neighbors = m_neigh;

        value_type HbetaOut[8];
        value_type BbetaOut[8];
        value_type HbetaIn[8];
        value_type BbetaIn[8];
        value_type HTotOut = 0.0;
        value_type BTotOut = 0.0;
        value_type HIn = 0.0;
        value_type BIn = 0.0;        

        for (int k=0; k<8; k++) {            
            HbetaOut[k] = m_para.HbetaOut[k];
            HbetaIn[k] = m_para.HbetaIn[k];
            BbetaOut[k] = m_para.BbetaOut[k];
            BbetaIn[k] = m_para.BbetaIn[k];
            HTotOut += HbetaOut[k];
            BTotOut += BbetaOut[k];
        }
        for (int k=0; k<8; k++) {
            HIn += HbetaIn[k] * neighbors.Ah[k];
            BIn += BbetaIn[k] * neighbors.Ao[k];
        }
        HIn = 0.0;
        BIn = 0.0;
        HTotOut = 0.0;
        BTotOut = 0.0;

        f[0] = b * psi_B * rho_Ao * x[5] - (mu_E + rho_E) * x[0];
        f[1] = rho_E * x[0] - (mu_L1 + mu_L2*x[1] + rho_L) * x[1];
        f[2] = rho_L * x[1] - (mu_P + rho_P) * x[2];
        f[3] = rho_P * x[2] + psi_B * rho_Ao * x[5] - (mu_Ah + psi_H * rho_Ah) * x[3] - HTotOut * x[3] + HIn;
        f[4] = psi_H * rho_Ah * x[3] - (mu_Ar + rho_Ar) * x[4];
        f[5] = rho_Ar * x[4] - (mu_Ao + psi_B * rho_Ao) * x[5] - BTotOut * x[5] + BIn;
    }
};



template<size_t N> struct MalariaDeterministic
{
    ParameterClassMalaria m_para;

    public:
    MalariaDeterministic( ParameterClassMalaria para ) : m_para(para) { }
    typedef boost::array<value_type, N> state_type;

    void operator() ( const state_type &x , state_type &f) const
    {
        value_type Ah = m_para.Ah;
        value_type psi_h = m_para.psi_h;
        value_type psi_v = m_para.psi_v;
        value_type sigma_v = m_para.sigma_v;
        value_type sigma_h = m_para.sigma_h;
        value_type beta_hv = m_para.beta_hv;
        value_type beta_vh = m_para.beta_vh;
        value_type beta_til_vh = m_para.beta_til_vh;
        value_type nu_h = m_para.nu_h;
        value_type nu_v = m_para.nu_v;
        value_type gamma_h = m_para.gamma_h;
        value_type delta_h = m_para.delta_h;
        value_type rho_h = m_para.rho_h;
        value_type mu_1h = m_para.mu_1h;
        value_type mu_2h = m_para.mu_2h;
        value_type mu_1v = m_para.mu_1v;
        value_type mu_2v = m_para.mu_2v;
        
        value_type Nh = x[0] + x[1] + x[2] + x[3];
        value_type Nv = x[4] + x[5] + x[6];
        value_type fh = mu_1h+mu_2h*Nh;
        value_type fv = mu_1v+mu_2v*Nv;
        value_type lhv = sigma_v*sigma_h / (sigma_v*Nv+sigma_h*Nh);

        f[0] = Ah + psi_h*Nh + rho_h*x[3] - lhv*(beta_hv*x[6])*x[0]-fh*x[0];
        f[1] = lhv * (beta_hv*x[6]) * x[0] - nu_h*x[1] - fh*x[1];
        f[2] = nu_h*x[1] - gamma_h*x[2] - fh*x[2] - delta_h*x[2];
        f[3] = gamma_h*x[2] - rho_h*x[3] - fh*x[3];
        f[4] = psi_v*Nv - lhv * (beta_vh*x[2] + beta_til_vh*x[3])*x[4] - fv*x[4];
        f[5] = lhv * (beta_vh*x[2] + beta_til_vh*x[3])*x[4] - nu_v*x[5] - fv*x[5];
        f[6] = nu_v*x[5] - fv*x[6];
    }
};


struct ELPA_Wiener
{   
    ParameterClassVector m_para;
    boost::array<value_type,6> m_xv;

    public:
    ELPA_Wiener( ParameterClassVector para, boost::array<value_type,6> xv ) : m_para(para), m_xv(xv) { }
    typedef boost::array< value_type , 6 > state_type; 
    void operator()( const state_type &x , state_type &f )
    {
        value_type b = m_para.b;
        value_type rho_E = m_para.rho_E;
        value_type rho_L = m_para.rho_L;
        value_type rho_P = m_para.rho_P;
        value_type rho_Ah = m_para.rho_Ah;
        value_type rho_Ar = m_para.rho_Ar;
        value_type rho_Ao = m_para.rho_Ao;
        value_type mu_E = m_para.mu_E;
        value_type mu_L1 = m_para.mu_L1;
        value_type mu_L2 = m_para.mu_L2;
        value_type mu_P = m_para.mu_P;
        value_type mu_Ah = m_para.mu_Ah;
        value_type mu_Ar = m_para.mu_Ar;
        value_type mu_Ao = m_para.mu_Ao;
        value_type psi_H = m_para.psi_H;
        value_type psi_B = m_para.psi_B;

        value_type E = m_xv[0];
        value_type L = m_xv[1];
        value_type P = m_xv[2];
        value_type Ah = m_xv[3];
        value_type Ar = m_xv[4];
        value_type Ao = m_xv[5];        

        value_type p1, p2, p3, p4, p5, p6, p7;
        value_type p8, p9, p10, p11, p12, p13;

        p1 = b * psi_B * rho_Ao * Ao;
        p2 = mu_E * E;
        p3 = rho_E * E;
        p4 = (mu_L1 + mu_L2*L) * L;
        p5 = rho_L * L;
        p6 = mu_P * P;
        p7 = rho_P * P;
        p8 = psi_B * rho_Ao * Ao;
        p9 = mu_Ah * Ah; 
        p10 = psi_H * rho_Ah * Ah;
        p11 = mu_Ar * Ar;
        p12 = rho_Ar * Ar;
        p13 = mu_Ao * Ao;

        struct timeval tim;
        gettimeofday(&tim, NULL);
        value_type ti = tim.tv_sec+(tim.tv_usec);   
        RNGType rng(ti);
        boost::normal_distribution<> rdist(0.0,1.0);
        boost::variate_generator< RNGType, boost::normal_distribution<> > get_rand(rng, rdist);
        std::vector<value_type> W(13);
        generate(W.begin(), W.end(), get_rand);
        f[0] = sqrt(p1)*W[0] - sqrt(p2)*W[1] - sqrt(p3)*W[2];
        f[1] = sqrt(p3)*W[2] - sqrt(p4)*W[3] - sqrt(p5)*W[4];
        f[2] = sqrt(p5)*W[4] - sqrt(p6)*W[5] - sqrt(p7)*W[6];
        f[3] = sqrt(p7)*W[6] + sqrt(p8)*W[7] - sqrt(p9)*W[8] - sqrt(p10)*W[9];
        f[4] = sqrt(p10)*W[9] - sqrt(p11)*W[10] - sqrt(p12)*W[11];
        f[5] = -sqrt(p8)*W[7] + sqrt(p12)*W[11] - sqrt(p13)*W[12];
    };
};


struct SEIR_Wiener
{   
    ParameterClassMalaria m_para;
    boost::array<value_type,7> m_xm;

    public:
    SEIR_Wiener( ParameterClassMalaria para, boost::array<value_type,7> xm ) : m_para(para), m_xm(xm) { }
    typedef boost::array< value_type , 7 > state_type; 
    void operator()( const state_type &x , state_type &f )
    {
        value_type Ah = m_para.Ah;
        value_type psi_h = m_para.psi_h;
        value_type psi_v = m_para.psi_v;
        value_type sigma_v = m_para.sigma_v;
        value_type sigma_h = m_para.sigma_h;
        value_type beta_hv = m_para.beta_hv;
        value_type beta_vh = m_para.beta_vh;
        value_type beta_til_vh = m_para.beta_til_vh;
        value_type nu_h = m_para.nu_h;
        value_type nu_v = m_para.nu_v;
        value_type gamma_h = m_para.gamma_h;
        value_type delta_h = m_para.delta_h;
        value_type rho_h = m_para.rho_h;
        value_type mu_1h = m_para.mu_1h;
        value_type mu_2h = m_para.mu_2h;
        value_type mu_1v = m_para.mu_1v;
        value_type mu_2v = m_para.mu_2v;
        
        value_type Sh = m_xm[0];
        value_type Eh = m_xm[1];
        value_type Ih = m_xm[2];
        value_type Rh = m_xm[3];
        value_type Nh = Sh + Eh + Ih + Rh;
        
        value_type Sv = m_xm[4];
        value_type Ev = m_xm[5];
        value_type Iv = m_xm[6];
        value_type Nv = Sv + Ev + Iv;

        value_type fh = mu_1h+mu_2h*Nh;
        value_type fv = mu_1v+mu_2v*Nv;
        value_type lhv = sigma_v*sigma_h / (sigma_v*Nv+sigma_h*Nh);
        value_type p1, p2, p3, p4, p5, p6, p7, p8;
        value_type p9, p10, p11, p12, p13, p14, p15;

        p1 = Ah + psi_h * Nh;
        p2 = rho_h * R;
        p3 = lhv * beta_hv * Iv * Sh;
        p4 = fh * Sh;
        p5 = nu_h * Eh;
        p6 = fh * Eh;
        p7 = gamma_h * Ih;
        p8 = (fh + delta_h) * Ih;
        p9 = fh * Rh;
        p10 = psi_v * Nv;
        p11 = lhv * (beta_vh*Ih + beta_til_vh*Rh);
        p12 = fv * Sv;
        p13 = nu_v * Ev;
        p14 = fv * Ev;
        p15 = fv * Iv;

        struct timeval tim;
        gettimeofday(&tim, NULL);
        value_type ti = tim.tv_sec+(tim.tv_usec);   
        RNGType rng(ti);
        boost::normal_distribution<> rdist(0.0,1.0);
        boost::variate_generator< RNGType, boost::normal_distribution<> > get_rand(rng, rdist);
        std::vector<value_type> W(15);
        generate(W.begin(), W.end(), get_rand);
        f[0] = sqrt(p1)*W[0] + sqrt(p2)*W[1] - sqrt(p3)*W[2] - sqrt(p4)*W[3];
        f[1] = sqrt(p3)*W[2] - sqrt(p5)*W[4] - sqrt(p6)*W[5];
        f[2] = sqrt(p5)*W[4] - sqrt(p7)*W[6] - sqrt(p8)*W[7];
        f[3] = sqrt(p2)*W[1] + sqrt(p7)*W[6] - sqrt(p9)*W[8];
        f[4] = sqrt(p10)*W[9] - sqrt(p11)*W[10] - sqrt(p12)*W[11];
        f[5] = sqrt(p11)*W[10] - sqrt(p13)*W[12] - sqrt(p14)*W[13];
        f[6] = -sqrt(p15)*W[14];
    };
};



// --------------------------------------------------------------------
// rand_gen()
//    Generate random number ranging from a to b
// --------------------------------------------------------------------
value_type rand_gen(value_type a, value_type b) 
{
    if (a > b) return b;
    return a + (value_type)(rand())/((value_type)(RAND_MAX/(b - a)));
}


// --------------------------------------------------------------------
// get_time()
//    Get current computer time
// --------------------------------------------------------------------
double get_time()
{
  struct timeval tim;
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}


// --------------------------------------------------------------------
// GetFileInfo()
//    Get information and dimension in NetCDF files
// --------------------------------------------------------------------
void GetFileInfo(const char *file_name, // File name 
        const char *var_name,   // Name of variable
        int ndims,              // Number of dimension
        int *dim)               // Dimension info
{
    int ncid, var_id, status, dimids[ndims];
    size_t length;

    // Open Netcdf file with NC_NOWRITE options
    status = nc_open(file_name, NC_NOWRITE, &ncid);
    if (status != NC_NOERR) {
        printf("ERROR: Line (%d) of Function <%s>. \n", __LINE__, __func__);
        printf("File '%s' not found! Exit now... \n", file_name);
        exit(1);
    }

    // Get variable id, dimension, size
    nc_inq_varid(ncid, var_name, &var_id);
    nc_inq_vardimid(ncid, var_id, dimids);

    for (int i = 0; i < ndims; i++) {
        nc_inq_dimlen(ncid, dimids[i], &length);
        dim[i] = length;
    }

    // Close Netcdf file
    nc_close(ncid);
}


// --------------------------------------------------------------------
// LoadFile()
//    Load data from NetCDF file to memory
// --------------------------------------------------------------------
template<typename Type> void LoadFile (
        const char *file_name,      // File name 
        const char *var_name,       // Name of variable
        Type *data )                // Data assigned
{
    int ncid, varid, status;          // local NetCDF vars

    // Open Netcdf file with NC_NOWRITE options (for Loading)
    status = nc_open(file_name, NC_NOWRITE, &ncid);
    if (status != NC_NOERR) {
        printf("ERROR: Line (%d) of Function <%s>. \n", __LINE__, __func__);
        printf("File '%s' not found! Exit now... \n", file_name);
        exit(1);
    }

    nc_inq_varid(ncid, var_name, &varid);     // Get variable ID
    nc_get_var(ncid, varid, &data[0]);        // Get data based on data_type
    nc_close(ncid);                           // Close the NetCDF file
}


// --------------------------------------------------------------------
// SaveOutput2D()
//    Saves 2D variables into NetCDF files.
// --------------------------------------------------------------------
template<typename Type> void SaveOutput2D(
        const char *filename,       // File name 
        int My, int Nx,             // 2D dimension of variable
        const char *var_name,       // Name of variable
        Type *data,                 // Data saved
        nc_type NC_DATATYPE,        // Type of data
        int write)                  // Write option: new or append
{
    int ncid, x_dimid, y_dimid, varid, dimids[2];

    // Set up NetCDF file for writing
    if (write == 0){      // Create a new NetCDF file
        nc_create(filename, NC_CLOBBER, &ncid);
    } else {              // Open and re-define an existing NetCDF file
        nc_open(filename, NC_WRITE, &ncid);
        nc_redef(ncid);
    }

    nc_def_dim(ncid, "x", Nx, &x_dimid);
    nc_def_dim(ncid, "y", My, &y_dimid);
    dimids[0] = y_dimid;
    dimids[1] = x_dimid;    

    nc_def_var(ncid, var_name, NC_DATATYPE, 2, dimids, &varid);
    nc_enddef(ncid);
    nc_put_var(ncid, varid, &data[0]);
    nc_close(ncid);
}


// --------------------------------------------------------------------
// SaveOutput3D()
//    Saves 3D variables into NetCDF files.
// --------------------------------------------------------------------
template<typename Type> void SaveOutput3D(
        const char *file,           // File name 
        int My, int Nx, int Pz,     // 2D dimension of variable
        const char *var_name,       // Name of variable
        Type *data,                 // Data saved
        nc_type NC_DATATYPE,        // Type of data
        int write)                  // Write option: new or append
{
  int ncid, x_dimid, y_dimid, z_dimid, varid, dimids[3];

  // Set up NetCDF file for writing
    if (write == 0){    // Create a new NetCDF file
        nc_create(file, NC_CLOBBER, &ncid);
    } else {            // Open and re-define an existing NetCDF file
        nc_open(file, NC_WRITE, &ncid);
        nc_redef(ncid);
    }

    nc_def_dim(ncid, "x", Nx, &x_dimid);
    nc_def_dim(ncid, "y", My, &y_dimid);
    nc_def_dim(ncid, "z", Pz, &z_dimid);

    dimids[0] = y_dimid;
    dimids[1] = x_dimid;
    dimids[2] = z_dimid;

    nc_def_var(ncid, var_name, NC_DATATYPE, 3, dimids, &varid);
    nc_enddef(ncid);
    nc_put_var(ncid, varid, &data[0]);
    nc_close(ncid);
}



// --------------------------------------------------------------------
// EstimateDevRate()
//    Estimate Develoment Rate as function air temperature.
// --------------------------------------------------------------------
value_type EstimateDevRate(ParameterClassDepinay *RateDevTemp, value_type TaC, int ind)
{
    value_type TaK = TaC + 273.15;
    value_type rT;
    value_type rho25C = RateDevTemp[ind].rho25C;
    value_type EtpA = RateDevTemp[ind].EtpA;
    value_type EtpL = RateDevTemp[ind].EtpL;
    value_type EtpH = RateDevTemp[ind].EtpH;
    value_type ThalfL = RateDevTemp[ind].ThalfL;
    value_type ThalfH = RateDevTemp[ind].ThalfH;

    rT = rho25C * (TaK/298.) * exp(EtpA/R * (1./298. - 1./TaK)) / 
         ( 1. + exp(EtpL/R * (1./ThalfL - 1./TaK)) + exp(EtpH/R * (1./ThalfH - 1./TaK)) );

    return rT;
} 



// --------------------------------------------------------------------
// EstimateInfectionRate()
//    Estimate Infection Rate inside vector as function air temperature.
// --------------------------------------------------------------------
value_type EstimateInfectionRate(value_type TaC)
{
    value_type rT;
    if (TaC > 15.5 && TaC < 35) {
        rT = 0.000112 * TaC * (TaC - 15.384) * sqrt(35. - TaC) / 24.;
    } else {
        rT = 1.0e-4;
    }
    return rT;
} 


// --------------------------------------------------------------------
// SetUpParametersVectors()
//    Set the parameters for Vector dynamics model
// --------------------------------------------------------------------
void SetUpParametersVectors(ParameterClassVector *pars_vec, ParameterClassMalaria 
        *pars_mal, ParameterClassDepinay *RateDevTemp, value_type *Hmap, value_type 
        *Bmap, int *counts, int *displ, int num_pts, int offset, int M, int N) 
{
    int glob_ind;
    for (int lind=0; lind<num_pts; lind++) {
        glob_ind = lind + offset;

        // Vector model parameters. Unit is [1/hr] unless specified
        pars_vec[lind].b = 50;             // [-]
        pars_vec[lind].rho_E = 0.33/24.;
        pars_vec[lind].rho_L = 0.08/24.;
        pars_vec[lind].rho_P = 0.33/24.;
        pars_vec[lind].rho_Ah = 0.46/24.;
        pars_vec[lind].rho_Ar = 0.43/24.;
        pars_vec[lind].rho_Ao = 3.0/24.;
        pars_vec[lind].mu_E = 0.056/24.;
        pars_vec[lind].mu_L1 = 0.44/24.;
        pars_vec[lind].mu_L2 = 0.05/24.;        // [1/(hr*mosp)]
        pars_vec[lind].mu_P = 0.37/24.;
        pars_vec[lind].mu_Ah = 0.18/24.;
        pars_vec[lind].mu_Ar = 0.0043/24.;
        pars_vec[lind].mu_Ao = 0.41/24.;
        pars_vec[lind].diff = 0.01;
        pars_vec[lind].lambda = 0.5;        // [-]  
        for (int k=0; k<8; k++) {
            pars_vec[lind].HbetaOut[k] = 1.0;
            pars_vec[lind].BbetaOut[k] = 1.0;
            pars_vec[lind].HbetaIn[k] = 1.0;
            pars_vec[lind].BbetaIn[k] = 1.0;
        }
        if (Hmap[glob_ind] > 0){
            pars_vec[lind].psi_H = 1.0;
        } else {
            pars_vec[lind].psi_H = 0.1;
        }
        if (Bmap[glob_ind] > 1){
            pars_vec[lind].psi_B = 1.0;
        } else {
            pars_vec[lind].psi_B = 0.1;
        }

        // Malaria model parameters. Unit is [1/hr] unless specified
        // Two birth rate parameters for mosquitoes is set to 0
        // The birth rate in mosquitoes is simulated in Vector model
        pars_mal[lind].Ah = 1.375e-3;
        pars_mal[lind].psi_h = 4.58e-6;         // human birth
        pars_mal[lind].psi_v = 5.42e-3;         // vector birth
        pars_mal[lind].sigma_v = 2.08e-2;
        pars_mal[lind].sigma_h = 0.792;
        pars_mal[lind].beta_hv = 0.022;
        pars_mal[lind].beta_vh = 0.48;
        pars_mal[lind].beta_til_vh = 0.048;
        pars_mal[lind].nu_h = 4.16e-3;
        pars_mal[lind].nu_v = 3.79e-3;
        pars_mal[lind].gamma_h = 8.46e-4;
        pars_mal[lind].delta_h = 3.75e-6;
        pars_mal[lind].rho_h = 2.3e-5;
        pars_mal[lind].mu_1h = 6.66e-7;         // human mortality
        pars_mal[lind].mu_2h = 1.25e-8;         // human mortality

        // Two rate of death parameters below are set to 0
        // The dying rates in mosquitoes are simulated in Vector model
        pars_mal[lind].mu_1v = 1.38e-3;
        pars_mal[lind].mu_2v = 8.3e-7;
    }

    // Egg stage
    RateDevTemp[0].rho25C = 0.0413;
    RateDevTemp[0].EtpA = 1.0;
    RateDevTemp[0].EtpL = -170644;
    RateDevTemp[0].EtpH = 1e6;
    RateDevTemp[0].ThalfL = 288.8;
    RateDevTemp[0].ThalfH = 313.3;

    // Larvae stage
    RateDevTemp[1].rho25C = 0.0037;
    RateDevTemp[1].EtpA = 15684;
    RateDevTemp[1].EtpL = -229902;
    RateDevTemp[1].EtpH = 822285;
    RateDevTemp[1].ThalfL = 286.4;
    RateDevTemp[1].ThalfH = 310.3;

    // Pupae stage
    RateDevTemp[2].rho25C = 0.034;
    RateDevTemp[2].EtpA = 1.0;
    RateDevTemp[2].EtpL = -154394;
    RateDevTemp[2].EtpH = 554707;
    RateDevTemp[2].ThalfL = 289.0;
    RateDevTemp[2].ThalfH = 313.8;

    // Pupae stage
    RateDevTemp[3].rho25C = 0.02;
    RateDevTemp[3].EtpA = 1000.0;
    RateDevTemp[3].EtpL = -75371;
    RateDevTemp[3].EtpH = 388691;
    RateDevTemp[3].ThalfL = 293.1;
    RateDevTemp[3].ThalfH = 313.4;
}


// --------------------------------------------------------------------
// UpdateParameters()
// --------------------------------------------------------------------
void UpdateParameters(ParameterClassVector *paras_vec, ParameterClassMalaria 
        *paras_mal, ParameterClassDepinay *RateDevTemp, value_type *TaC, 
        value_type dt, int num_pts, int offset, int rank, int M, int N, int t) 
{
    int glob_ind;
    value_type d, rT;

    for (int lind=0; lind<num_pts; lind++) {
        glob_ind = lind + offset;
        // Vector model parameters. Unit is [1/hr] unless specified
        paras_vec[lind].rho_E = EstimateDevRate(RateDevTemp, TaC[t], 0) * dt;
        paras_vec[lind].rho_L = EstimateDevRate(RateDevTemp, TaC[t], 1) * dt;
        paras_vec[lind].rho_P = EstimateDevRate(RateDevTemp, TaC[t], 2) * dt;
        paras_vec[lind].rho_Ah = EstimateDevRate(RateDevTemp, TaC[t], 3) * dt;
        paras_vec[lind].rho_Ar = EstimateDevRate(RateDevTemp, TaC[t], 3) * dt;
        paras_vec[lind].rho_Ao = EstimateDevRate(RateDevTemp, TaC[t], 3) * dt;
        paras_mal[lind].nu_v = EstimateInfectionRate(TaC[t]);
    }    
}


// --------------------------------------------------------------------
// GetNeighborLocation()
//    Identify the location of the 8 neighbors around the cells
//    Period boundary conditions are used at the edges
// --------------------------------------------------------------------
void GetNeighborLocation(int *neighbors, int i, int j, int M, int N)
{
    int iw, ie, jn, js;

    // West side bound
    if (i == 0) {
        iw = N-1;
    } else {
        iw = i-1;
    }
    neighbors[4] = j*N + iw;

    // East side bound
    if (i == N-1) {
        ie = 0;
    } else {
        ie = i+1;
    }
    neighbors[0] = j*N + ie;

    // North side bound
    if (j == M-1) {
        jn = 0;
    } else {
        jn = j+1;
    }
    neighbors[6] = jn*N + i;

    // South side bound
    if (j == 0) {
        js = M-1;
    } else {
        js = j-1;
    }
    neighbors[2] = js*N + i;

    // North West bound
    if (i==0 && j==M-1) {
        neighbors[5] = 0;
    } else {
        neighbors[5] = jn*N + iw;
    }

    // North East bound
    if (i==N-1 && j==M-1) {
        neighbors[7] = 0;
    } else {
        neighbors[7] = jn*N + ie;
    }

    // South West bound
    if (i==0 && j==0) {
        neighbors[3] = 0;
    } else {
        neighbors[3] = js*N + iw;
    }

    // South East bound
    if (i==N-1 && j==0) {
        neighbors[1] = 0;
    } else {
        neighbors[1] = js*N + ie;
    }
}


// --------------------------------------------------------------------
// GetNeighborQuantity()
//    Estimate the population of 8 neighbor around the cells
//    Period boundary conditions are used at the edges
// --------------------------------------------------------------------
void GetNeighborQuantity(value_type *neighbors, value_type *data, int i, int j,
        int M, int N)
{
    int iw, ie, jn, js;

    // West side bound
    if (i == 0) {
        iw = N-1;
    } else {
        iw = i-1;
    }
    neighbors[4] = data[j*N + iw];

    // East side bound
    if (i == N-1) {
        ie = 0;
    } else {
        ie = i+1;
    }
    neighbors[0] = data[j*N + ie];

    // North side bound
    if (j == M-1) {
        jn = 0;
    } else {
        jn = j+1;
    }
    neighbors[6] = data[jn*N + i];

    // South side bound
    if (j == 0) {
        js = M-1;
    } else {
        js = j-1;
    }
    neighbors[2] = data[js*N + i];

    // North West bound
    if (i==0 && j==M-1) {
        neighbors[5] = 0;
    } else {
        neighbors[5] = data[jn*N + iw];
    }

    // North East bound
    if (i==N-1 && j==M-1) {
        neighbors[7] = 0;
    } else {
        neighbors[7] = data[jn*N + ie];
    }

    // South West bound
    if (i==0 && j==0) {
        neighbors[3] = 0;
    } else {
        neighbors[3] = data[js*N + iw];
    }

    // South East bound
    if (i==N-1 && j==0) {
        neighbors[1] = 0;
    } else {
        neighbors[1] = data[js*N + ie];
    }
}


// --------------------------------------------------------------------
// GetD8StatisticNeighborCells()
//    Find neighbors and estimate the density of data in D8 ring
// --------------------------------------------------------------------
void GetD8StatisticNeighborCells(value_type *data, StatisticsClass *stat,
        int num_pts, int offset, int M, int N)
{
    int i, j, glob_ind;
    value_type neighbors[9];

    for (int lind=0; lind<num_pts; lind++) {
        glob_ind = lind + offset;
        i = glob_ind % N;
        j = glob_ind / N;
        neighbors[8] = data[glob_ind];

        // Get the number of neighbors around the cell.
        GetNeighborQuantity(neighbors, data, i, j, M, N);

        // Estimate the total host number and proportion (0-1)
        stat[lind].total = 0.0;
        for (int k=0; k<9; k++){
            stat[lind].total += neighbors[k];
        }

        if (stat[lind].total == 0) {
            for (int k=0; k<8; k++){
                stat[lind].D8[k] = 0.0;                    
            }
            stat[lind].center = 0.0;
        } else {
            for (int k=0; k<8; k++){      
                stat[lind].D8[k] = (value_type) neighbors[k]/stat[lind].total;
            }
            stat[lind].center = (value_type) neighbors[8]/stat[lind].total;
        }
    }
}


// --------------------------------------------------------------------
// EstimateRateOutofCells()
//    Calculate the move-out rate from center cell to 8 neighbors
// --------------------------------------------------------------------
void EstimateRatesOutofCells(StatisticsClass *Hstat, StatisticsClass *Bstat, 
        ParameterClassVector *pars_V, int num_pts, int offset, int rank, int M,
        int N)
{
    int i, j, glob_ind;
    value_type D, lambda;

    for (int lind=0; lind<num_pts; lind++) {
        glob_ind = lind + offset;
        i = glob_ind % N;
        j = glob_ind / N;  
        
        D = pars_V[lind].diff;
        lambda = pars_V[lind].lambda;
        for (int k=0; k<8; k++) {
            pars_V[lind].HbetaOut[k] = D * exp(-lambda * (Hstat[lind].center-Hstat[lind].D8[k]));
            pars_V[lind].BbetaOut[k] = D * exp(-lambda * (Bstat[lind].center-Bstat[lind].D8[k]));
        }
    }
}


// --------------------------------------------------------------------
// EstimateRateIntoCells()
//    Calculate the move-in rate from 8 neighbors into center cells
// --------------------------------------------------------------------
void EstimateRateIntoCells(StatisticsClass *Hstat, StatisticsClass *Bstat,
        StatisticsClass *HstatMN, StatisticsClass *BstatMN, ParameterClassVector 
        *pars_V, int num_pts, int offset, int rank, int M, int N)
{
    int i, j, glob_ind, m, ind;
    value_type D, lambda;
    int neigh_positions[8];

    for (int lind=0; lind<num_pts; lind++) {
        glob_ind = lind + offset;
        i = glob_ind % N;
        j = glob_ind / N;  
  
        GetNeighborLocation(neigh_positions, i, j, M, N);           
        for (int k=0; k<8; k++) {
            m = (k < 4) ? k+4 : k-4;
            ind = neigh_positions[k];                
            pars_V[lind].HbetaIn[k] = D * exp( -lambda*(Hstat[lind].center-HstatMN[ind].D8[m]) );
            pars_V[lind].BbetaIn[k] = D * exp( -lambda*(Bstat[lind].center-BstatMN[ind].D8[m]) );
        }
    }
}


// --------------------------------------------------------------------
// SetUpInitialConditions()
//    Set up initial conditions for vector and malaria models.
// --------------------------------------------------------------------
void SetUpInitialConditions(value_type *XVec, value_type *XMal, 
        value_type *XVecGlob, value_type *XMalGlob, value_type *Hmap, 
        value_type *Bmap, int *counts, int *displ, int *counts6, int *displs6, 
        int *counts7, int *displs7, int num_pts, int offset, int M, int N)
{
    int glob_ind;
    value_type E, L, P, Ah, Ar, Ao;
    value_type Nh, Sh, Eh, Ih, Rh, Nv, Sv, Ev, Iv;
    E = 270.0; L = 190.0; P = 200.0;
    Ah = 240.0; Ar = 180.0; Ao = 120.0;
    value_type f = 0.01;
    // Initialization in local pid
    for (int lind=0; lind<num_pts; lind++) {
        glob_ind = lind + offset;
 
        // VECTOR POPULATIONS
        XVec[lind*6+0] = E * f;// * rand_gen(0.3, 1.0);
        XVec[lind*6+1] = L * f;// * rand_gen(0.3, 1.0);
        XVec[lind*6+2] = P * f;// * rand_gen(0.3, 1.0);
        XVec[lind*6+3] = Ah * f;// * rand_gen(0.1, 1.0);
        XVec[lind*6+4] = Ar * f;// * rand_gen(0.1, 1.0);
        XVec[lind*6+5] = Ao * f;// * rand_gen(0.1, 1.0);
        
        // MALARIA POPULATIONS            
        // Host human sub-groups
        Nh = Hmap[glob_ind];
        XMal[lind*7+0] = 500.;  //rand_gen(0.1, 0.2);    // Sh
        XMal[lind*7+1] = 10;    //rand_gen(0.01, 0.2);   // Eh
        XMal[lind*7+2] = 30.;   //rand_gen(0.01, 0.2);   // Ih
        XMal[lind*7+3] = 0.;    //Nh;                    // Rh
        
        // Vector sub-groups
        Nv = 4150.; //XVec[lind*6 + 3] + XVec[lind*6 + 4] + XVec[lind*6 + 5];
        XMal[lind*7+4] = 4000.; //rand_gen(0.01, 0.1);  // Sv
        XMal[lind*7+5] = 100.;  //rand_gen(0.01, 0.1);  // Ev
        XMal[lind*7+6] = 50;    //                      // Iv
    }
    MPI_Barrier( MPI_COMM_WORLD );      // Sync all process

    // Gather data to master pid and Bcast to all
    MPI_Gatherv(XVec, num_pts*6, mpi_type, XVecGlob, counts6, displs6, mpi_type, 0, MPI_COMM_WORLD);
    MPI_Gatherv(XMal, num_pts*7, mpi_type, XMalGlob, counts7, displs7, mpi_type, 0, MPI_COMM_WORLD);
    MPI_Bcast(XVecGlob, 6*M*N, mpi_type, 0, MPI_COMM_WORLD);
    MPI_Bcast(XMalGlob, 7*M*N, mpi_type, 0, MPI_COMM_WORLD);
}


// --------------------------------------------------------------------
// ODEsVectorModel()
//    Set up and solve the vector dynamics model in device
// --------------------------------------------------------------------
void ODEsVectorModel(value_type *X_Vec, value_type *X_VecMN, 
        ParameterClassVector *paras_vec, int t, value_type dt, int num_pts, 
        int offset, int rank, int M, int N, int print_flag)
{
    int i, j, glob_ind;
    ParameterClassVector paras;
    NeighborsClass neighbors;
    boost::array<value_type,6> xv;
    value_type Vec_Ah[M*N];
    value_type Vec_Ao[M*N];

    // Transfer Ah, Ao to separate dataset
    for (int lind=0; lind<M*N; lind++) {
        Vec_Ah[lind] = X_VecMN[lind*6 + 3];
        Vec_Ao[lind] = X_VecMN[lind*6 + 5];
    }

    for (int lind=0; lind<num_pts; lind++) {
        glob_ind = lind + offset;
        i = glob_ind % N;
        j = glob_ind / N;

        for (int k=0; k<6; k++) {
            xv[k] = X_Vec[lind*6 + k];
        }
        paras = paras_vec[lind];

        // Get Ah and Ao neighbor
        GetNeighborQuantity(neighbors.Ah, Vec_Ah, i, j, M, N);
        GetNeighborQuantity(neighbors.Ao, Vec_Ao, i, j, M, N);

        // Integrate and solve the SDEs with functors
        integrate_const(SDE_Euler<6>(), 
            make_pair(VectorDeterministic<6>(paras, neighbors), ELPA_Wiener(paras,xv)),
            xv, t*dt, (t+1)*dt, dt);

        // Update solution to global array
        for (int k=0; k<6; k++) {
            if (xv[k] < 0) { xv[k] = 0.0; }                
            X_Vec[lind*6 + k] = xv[k];
        }
    }
}


// --------------------------------------------------------------------
// ODEsMalariaModel()
//    Set up and solve the malaria transmission model in device
// --------------------------------------------------------------------
void ODEsMalariaModel(value_type *X_Mal, ParameterClassMalaria *paras_mal, 
        value_type *Hmap, int t, value_type dt, int num_pts, int offset, 
        int rank, int M, int N, int print_flag)
{
    int i, j, glob_ind;
    ParameterClassMalaria paras;
    boost::array<value_type,7> xm;

    for (int lind=0; lind<num_pts; lind++) {
        glob_ind = lind + offset;
        i = glob_ind % N;
        j = glob_ind / N;        
        for (int k=0; k<7; k++) {
            xm[k] = X_Mal[lind*7 + k];
        }
        if (Hmap[glob_ind] > 0) {
            paras = paras_mal[lind];

            // Integrate and solve the SDEs with functors
            integrate_const( SDE_Euler<7>(), 
                make_pair(MalariaDeterministic<7>(paras),SEIR_Wiener(paras,xm)), 
                xm, t*dt, (t+1)*dt, dt);
        }
        // Update solution to global array
        for (int k=0; k<7; k++) {
            if (xm[k] < 0) { xm[k] = 0.0; }
            X_Mal[lind*7 + k] = xm[k];
        }
    }
}


// --------------------------------------------------------------------
// UpdateVectorMalariaPopulation()
//    This function update the vector population to malaria population
//    The number of adult mosquitoes in each cell is multiplied with
//    the fraction of malaria status (Exposed, Infected, Recovered, Total).
//    Note that, we use the fraction population ODEs in the malaria model,
//    so the fraction is updated.
// --------------------------------------------------------------------
void UpdateVectorMalariaPopulation(value_type *XVec, value_type * XMal, 
        int num_pts, int offset, int rank, int M, int N)
{
    int glob_ind, vec_ind, mal_ind;
    value_type Nv, sh, eh, ih;

    for (int lind=0; lind<num_pts; lind++) {
        glob_ind = lind + offset;
        vec_ind = lind * 6;
        mal_ind = lind * 7;
        Nv = XVec[mal_ind+4] + XVec[mal_ind+5] + XVec[mal_ind+6];
        // Fractions of population are assumed unchanged.
        sh = XMal[mal_ind + 4] / Nv;
        eh = XMal[mal_ind + 5] / Nv;
        ih = XMal[mal_ind + 5] / Nv;
        Nv = XVec[vec_ind+3] + XVec[vec_ind+4] + XVec[vec_ind+5];
        XMal[mal_ind+4] = sh * Nv;
        XMal[mal_ind+5] = eh * Nv;
        XMal[mal_ind+6] = ih * Nv;
    }
}


// --------------------------------------------------------------------
// GetTotalPopulation()   
// Sum all population in each group
// --------------------------------------------------------------------
void GetTotalPopulation(value_type *XVec, value_type *XVectotal, 
        value_type *XMal, value_type *XMaltotal, int *cnt6, int *cnt7,
        int *disp6, int *disp7, int t, int num_steps, int rank, int num_pts, 
        int offset, int numtasks, int M, int N)
{
    int glob_ind, vec_ind, mal_ind;
    value_type *Vec_local = new value_type[6];
    value_type *Mal_local = new value_type[7]; 
    value_type *Vec_global = new value_type[6*numtasks];     
    value_type *Mal_global = new value_type[7*numtasks];

    for (int k=0; k<6; k++) {
        Vec_local[k] = 0.0;
    }
    for (int k=0; k<7; k++) {
        Mal_local[k] = 0.0;
    }

    // Loop for all grid points
    for (int lind=0; lind<num_pts; lind++) {
        glob_ind = lind + offset;
        vec_ind = lind * 6;
        mal_ind = lind * 7;
        for (int k=0; k<6; k++) {
            Vec_local[k] += XVec[vec_ind+k];
        }
        for (int k=0; k<7; k++) {
            Mal_local[k] += XMal[mal_ind+k];
        }
    }
    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Gatherv(Vec_local, 6, mpi_type, Vec_global, cnt6, disp6, mpi_type, 0, MPI_COMM_WORLD);
    MPI_Gatherv(Mal_local, 7, mpi_type, Mal_global, cnt7, disp7, mpi_type, 0, MPI_COMM_WORLD);
    
    MPI_Barrier( MPI_COMM_WORLD );
    if (rank == 0) {
        for (int k=0; k<6; k++) {
            XVectotal[k*num_steps+t] = 0.0;
        }
        for (int k=0; k<6; k++) {
            XMaltotal[k*num_steps+t] = 0.0;
        }
        for (int id=0; id<numtasks; id++) {    
            for (int k=0; k<6; k++) {
                XVectotal[k*num_steps+t] += Vec_global[id*6+k];
            }
            for (int k=0; k<7; k++) {
                XMaltotal[k*num_steps+t] += Mal_global[id*7+k];
            }
        }
    }
    MPI_Barrier( MPI_COMM_WORLD );
    free(Vec_local); free(Mal_local); free(Vec_global); free(Mal_global);
}

// --------------------------------------------------------------------
// GetAllTimePopulation()
//    Sum all population in each group
// --------------------------------------------------------------------
void GetAllTimePopulation(value_type *XData, value_type *XDataTimeSpatial, 
    int t, int num_steps, int M, int N, int size)
{
    int glob_ind, vec_ind;

    // Loop for all grid points
    for (int k=0; k<size; k++) {
        for (int j=0; j<M; j++) {
            for (int i=0; i<N; i++) {
                glob_ind = j*N + i;
                vec_ind = t*M*N*size + k*M*N + glob_ind;
                XDataTimeSpatial[vec_ind] = XData[k*M*N+glob_ind];
            }
        }
    }
}


void SplitDomainToProcess (int rank, int numtasks, int *row, int *offset, 
        int *displs, int *counts, int *displs6, int *counts6, int *displs7, 
        int *counts7, int M, int N)
{
    int row0, averow, extra, dest, source, msgtype;
    int tag = 1;
    MPI_Status status;
    
    if (rank == 0) {
        averow = M*N / numtasks;
        extra = M*N % numtasks;
        *offset = 0;
        for (int i=0; i<numtasks; i++) {
            *row = (i < numtasks-extra) ? averow : averow+1;
            counts[i] = *row;
            counts6[i] = *row * 6;
            counts7[i] = *row * 7;
            if (i==0){
                row0 = *row;
                *offset = *offset + *row;
                displs[i] = 0;
                displs6[i] = 0;
                displs7[i] = 0;
            } else {
                dest = i;
                MPI_Send(offset, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                MPI_Send(row, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                *offset += *row;
                displs[i] = displs[i-1] + counts[i-1];
                displs6[i] = displs6[i-1] + counts6[i-1];
                displs7[i] = displs7[i-1] + counts7[i-1];
            }
        }
        *offset = 0;
        *row = row0;      
    }

    if (rank != 0) {
        source = 0;
        msgtype = tag;
        MPI_Recv(offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
        MPI_Recv(row, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


// --------------------------------------------------------------------
// main()
//    Main program for vector dynamics and malaria transmission models.
//    The spatial model at each grid point includes two components:
//      - Vector dynamics submodel: simulates aquatic and adult stages of the 
//        mosquitoes.
//      - Malaria dynamics submodel: simulates human and mosquitoes.
// --------------------------------------------------------------------
int main(int argc, char **argv) 
{
    int rank, numtasks, offset, num_pts;
    int M, N, pf_vec, pf_mal, sf_vec, sf_mal, num_steps = 1000;    
    int dim_human[2], dim_water[2], dim_forcing[1];
    value_type ts, te;
    value_type *XVec_SpaGlob, *XMal_SpaGlob, *XVec_Spatial, *XMal_Spatial;
    value_type *XVectotal, *XMaltotal;
    value_type dt = 3.0;                        // Time step of ODEs [hr]
    char humanfile[64], waterfile[64], fileforcing[64];
    char fileoutput2D[64], fileoutput3D[64];
    srand(time(NULL));
    ofstream print_time("logs.txt", ios::app);

    pf_mal = 0;             // Print to screen flag for malaria model
    pf_vec = 0;             // Print to screen flag for vector model
    sf_mal = 0;             // Save to file flag for malaria model
    sf_vec = 0;             // Save to file flag for vector model

    // Initialize MPI parallel environment. Each processor gets its id.
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &numtasks);
    printf("Processor %d of %d activated \n", rank, numtasks);
    MPI_Barrier( MPI_COMM_WORLD );  // Sync all process

    /* create a type for struct StatisticsClass */
    MPI_Datatype Statisticstype;    
    MPI_Datatype types[3] = {mpi_type, mpi_type, mpi_type};
    int blocklen[3] = {1,1,8};
    MPI_Aint disp[3];
    
    disp[0] = offsetof(StatisticsClass, total);
    disp[1] = offsetof(StatisticsClass, center);
    disp[2] = offsetof(StatisticsClass, D8);
    MPI_Type_create_struct(3, blocklen, disp, types, &Statisticstype);
    MPI_Type_commit(&Statisticstype);

    snprintf(humanfile, sizeof(char) * 64, "data/human_data.nc");
    snprintf(waterfile, sizeof(char) * 64, "data/water_data.nc");
    snprintf(fileforcing, sizeof(char) * 64, "data/forcings_Kilifi10yr.nc");

    // Load human map data to identify dimension
    GetFileInfo(humanfile, "human", 2, dim_human);
    GetFileInfo(waterfile, "water", 2, dim_water);
    GetFileInfo(fileforcing, "Ta_in", 1, dim_forcing);

    if (dim_human[0] != dim_water[0] || dim_human[1] != dim_water[1]) {
        cout << "Human and water map dimension do NOT match!!!" << endl;
        exit(1);
    } else {
        M = dim_human[0];
        N = dim_human[1];
    }
    M = 5;
    N = 1;

    // Decompose domain into sub-domains for parallel computing.
    // Criteria to split the domain is up to users.
    int *displs = new int[numtasks];
    int *displs6 = new int[numtasks];
    int *displs7 = new int[numtasks];
    int *counts = new int[numtasks];
    int *counts6 = new int[numtasks];
    int *counts7 = new int[numtasks];
    int *cntGTP6 = new int[numtasks];
    int *cntGTP7 = new int[numtasks];
    int *dispGTP6 = new int[numtasks];
    int *dispGTP7 = new int[numtasks];
    for (int k=0; k<numtasks; k++) {
        cntGTP6[k] = 6;
        cntGTP7[k] = 7;
        dispGTP6[k] = k*6;
        dispGTP7[k] = k*7;
    }
    SplitDomainToProcess(rank, numtasks, &num_pts, &offset, displs, counts, 
            displs6, counts6, displs7, counts7, M, N);
    MPI_Barrier( MPI_COMM_WORLD );          // Sync all process

    // Global variables and parameters
    int numsteps_data = dim_forcing[0];
    
    // Spatial model has 6 unknowns, and Malaria model has 7 unknowns
    XVec_Spatial = new value_type[num_pts*6];           // num row x 6 col
    XMal_Spatial = new value_type[num_pts*7];           // num row x 7 col
    XVec_SpaGlob = new value_type[M*N*6];               // M*N row x 6 col
    XMal_SpaGlob = new value_type[M*N*7];               // M*N row x 7 col
    
    // Initialize global variables in master pid.
    XVectotal = new value_type[num_steps*6];            // num col x 6 row
    XMaltotal = new value_type[num_steps*7];            // num col x 7 row

    // Forcing variables
    value_type *Ta = new value_type[numsteps_data];
    value_type *PPT = new value_type[numsteps_data];
    value_type *ea = new value_type[numsteps_data];
    value_type *hour = new value_type[numsteps_data];
    value_type *decyear = new value_type[numsteps_data];

    // Two mapping variables
    value_type *Hmap = new value_type[M*N];
    value_type *Bmap = new value_type[M*N];

    // User-defined parameters and statistics
    ParameterClassVector *pars_vec = new ParameterClassVector[num_pts];
    ParameterClassMalaria *pars_mal = new ParameterClassMalaria[num_pts];
    StatisticsClass *Hstat = new StatisticsClass[num_pts];
    StatisticsClass *Bstat = new StatisticsClass[num_pts];
    ParameterClassDepinay *RateDevTemp = new ParameterClassDepinay[4];

    StatisticsClass *HstatMN = new StatisticsClass[M*N];
    StatisticsClass *BstatMN = new StatisticsClass[M*N];

    // Print out model info
    if (rank == 0) {
        printf("-----------------------------------------------\n");
        printf("              SPATIAL MALARIA MODEL            \n");
        printf("-----------------------------------------------\n");
        printf("Domain dimesion: %d x %d \n", M, N);
        printf("Time step: %5.2f [hr] \n", dt);
        printf("Forcing data length: %d \n", dim_forcing[0]);
        printf("# steps simulation: %d \n", num_steps);
        printf("-----------------------------------------------\n");
    }

    // Load human and static water map
    // LoadFile(humanfile, "human", Hmap);
    // LoadFile(waterfile, "water", Bmap);
    LoadFile(fileforcing, "Ta_in", Ta);
    LoadFile(fileforcing, "PPT_in", PPT);
    LoadFile(fileforcing, "ea_in", ea);
    LoadFile(fileforcing, "hour", hour);
    LoadFile(fileforcing, "decyear", decyear);
    for (int j=0; j<M; j++) {
        for (int i=0; i<N; i++) {
            Hmap[j*N+i] = 10. * rand_gen(0.01, 1.0);
            Bmap[j*N+i] = 2. * rand_gen(0.2, 1.0);
        }
    }
    for (int i=0; i<numsteps_data; i++) {
        Ta[i] += 0.0;
    }

for (int iloop=0; iloop<1; iloop++) {          
    // Set up initial conditions
    SetUpInitialConditions(XVec_Spatial, XMal_Spatial, XVec_SpaGlob, XMal_SpaGlob, Hmap, Bmap, counts, displs, counts6, displs6, counts7, displs7, num_pts, offset, M, N);

    if (rank == 0) {
        SaveOutput2D("results/Hmap.nc", M, N, "Hmap", Hmap, NC_TYPE, 0);
        SaveOutput2D("results/Bmap.nc", M, N, "Bmap", Bmap, NC_TYPE, 0);       
    }    

    // Set up parameters
    SetUpParametersVectors(pars_vec, pars_mal, RateDevTemp, Hmap, Bmap, counts, displs, num_pts, offset, M, N);

    // Estimate statistics info around every cell
    GetD8StatisticNeighborCells(Hmap, Hstat, num_pts, offset, M, N);
    GetD8StatisticNeighborCells(Bmap, Bstat, num_pts, offset, M, N);

    // Gather Statistics to master and broadcast to global
    MPI_Gatherv(Hstat, num_pts, Statisticstype, HstatMN, counts, displs, Statisticstype, 0, MPI_COMM_WORLD);
    MPI_Gatherv(Bstat, num_pts, Statisticstype, BstatMN, counts, displs, Statisticstype, 0, MPI_COMM_WORLD);
    MPI_Bcast(HstatMN, M*N, Statisticstype, 0, MPI_COMM_WORLD);
    MPI_Bcast(BstatMN, M*N, Statisticstype, 0, MPI_COMM_WORLD);

    ts = get_time();
    for (int t=0; t<num_steps; t++) {
        // Estimate movement rates
        // EstimateRatesOutofCells(Hstat, Bstat, pars_vec, num_pts, offset, rank, M, N);
        // MPI_Barrier( MPI_COMM_WORLD );        
        // EstimateRateIntoCells(Hstat, Bstat, HstatMN, BstatMN, pars_vec, num_pts, offset, rank, M, N);
        // MPI_Barrier( MPI_COMM_WORLD );
        
        // Update parameters
        // UpdateParameters(pars_vec, pars_mal, RateDevTemp, Ta, dt, num_pts, offset, rank, M, N, t);
        // MPI_Barrier( MPI_COMM_WORLD );

        // Run vector dynamics model
        ODEsVectorModel(XVec_Spatial, XVec_SpaGlob, pars_vec, t, dt, num_pts, offset, rank, M, N, pf_vec);
        MPI_Barrier( MPI_COMM_WORLD );

        // Update malaria population from vector dynamics
        // UpdateVectorMalariaPopulation(XVec_Spatial, XMal_Spatial, num_pts, offset, rank, M, N);
        // MPI_Barrier( MPI_COMM_WORLD );      

        // Run malaria dynamics model
        // ODEsMalariaModel(XMal_Spatial, pars_mal, Hmap, t, dt, num_pts, offset, rank, M, N, pf_mal);
        // MPI_Barrier( MPI_COMM_WORLD );            

        // Get total population each group
        GetTotalPopulation(XVec_Spatial, XVectotal, XMal_Spatial, XMaltotal, cntGTP6, cntGTP7, dispGTP6, dispGTP7, t, num_steps, rank, num_pts, offset, numtasks, M, N);
        MPI_Barrier( MPI_COMM_WORLD );

        // Save multi-dimensional output at specified steps
        if (t % 1000 == 0) {
            // Gather data to master for saving
            MPI_Gatherv(XVec_Spatial, num_pts*6, mpi_type, XVec_SpaGlob, counts6, displs6, mpi_type, 0, MPI_COMM_WORLD);
            MPI_Gatherv(XMal_Spatial, num_pts*7, mpi_type, XMal_SpaGlob, counts7, displs7, mpi_type, 0, MPI_COMM_WORLD);
            if (rank == 0) {            
                if (sf_vec == 1) {
                    snprintf(fileoutput3D, sizeof(char) * 64, "results/Vector3D_%d.nc", t);
                    SaveOutput3D(fileoutput3D, M, N, 6, "XVec", XVec_SpaGlob, NC_TYPE, 0);
                }
                if (sf_mal == 1) {
                    snprintf(fileoutput3D, sizeof(char) * 64, "results/Malaria3D_%d.nc", t);
                    SaveOutput3D(fileoutput3D, M, N, 7, "XMal", XMal_SpaGlob, NC_TYPE, 0);
                }
            }
        }    
        if (t % 10 == 0) {
            if (rank == 0) {            
                te = get_time();
                printf("Simulation %d of %d completed. Time = %10.5f\n", t, num_steps, te-ts);
                print_time<<"Simulation "<<t<<" of "<<num_steps<<" completed. Time="<<te-ts<< endl;
            }
        }
        MPI_Barrier( MPI_COMM_WORLD );
    }

    if (rank == 0) {
        snprintf(fileoutput2D, sizeof(char) * 64, "results/Total_malaria%d.nc",iloop);    
        //SaveOutput2D(fileoutput2D, 7, num_steps, "XMaltotal", XMaltotal, NC_TYPE, 0);
        snprintf(fileoutput2D, sizeof(char) * 64, "results/Total_vector5.nc");
        SaveOutput2D(fileoutput2D, 6, num_steps, "XVectotal", XVectotal, NC_TYPE, 0);    
        printf("Finalizing parallel session! Exit... \n");
    }
}
    print_time.close();
    MPI_Type_free(&Statisticstype);
    MPI_Finalize();
    free(XVec_Spatial); free(XMal_Spatial); free(XVec_SpaGlob); free(XMal_SpaGlob);
    free(XVectotal); free(XMaltotal); free(Ta); free(PPT); free(ea); 
    free(Hmap); free(Bmap); free(pars_vec); free(pars_mal); 
    free(Hstat); free(Bstat); free(RateDevTemp); free(HstatMN); free(BstatMN); 
    free(displs); free(displs6); free(displs7); 
    free(counts); free(counts6); free(counts7); 
    free(cntGTP6); free(cntGTP7); free(dispGTP6); free(dispGTP7);     
}