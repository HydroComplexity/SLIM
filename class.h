////////////////////////////////////////////////////////////
// value_type: Common type of variables/value defined
// in the main.cpp file.
////////////////////////////////////////////////////////////

#ifndef PARAMETERS_VECTOR_H
#define PARAMETERS_VECTOR_H
  // Mosquito class includes general parameters/options for mosquito agents
  class ParameterClassVector
  {
  public:
    // ODE model parameters
    value_type b;
    value_type rho_E;
    value_type rho_L;
    value_type rho_P;
    value_type rho_Ah;
    value_type rho_Ar;
    value_type rho_Ao;
    value_type mu_E;
    value_type mu_L1;
    value_type mu_L2;
    value_type mu_P;
    value_type mu_Ah;
    value_type mu_Ar;
    value_type mu_Ao;
    value_type psi_H;
    value_type psi_B;
    
    // Diffusion model parameters
    value_type diff;
    value_type lambda;
    value_type HbetaOut[8];
    value_type BbetaOut[8];
    value_type HbetaIn[8];
    value_type BbetaIn[8];
  };
#endif


#ifndef PARAMETERS_MALARIA_H
#define PARAMETERS_MALARIA_H
  // Mosquito class includes general parameters/options for mosquito agents
  class ParameterClassMalaria
  {
  public:
    value_type Ah;
    value_type psi_h;
    value_type psi_v;
    value_type sigma_v;
    value_type sigma_h;
    value_type beta_hv;
    value_type beta_vh;
    value_type beta_til_vh;
    value_type nu_h;
    value_type nu_v;
    value_type gamma_h;
    value_type delta_h;
    value_type rho_h;
    value_type mu_1h;
    value_type mu_2h;
    value_type mu_1v;
    value_type mu_2v;
  };
#endif


#ifndef STATISTICS_H
#define STATISTICS_H
  // Statistics class
  class StatisticsClass
  {
  public:
    value_type total;
    value_type center;
    value_type D8[8];
    // The the 8 neighbors is arranged as follow:
    // E:0, SE:1, S:2, SW:3, W:4, NW:5, N:6, NE:7
    // 5 6 7
    // 4 X 0
    // 3 2 1  
  };
#endif  


#ifndef NEIGHBORS_H
#define NEIGHBORS_H
  // Statistics class
  class NeighborsClass
  {
  public:
    value_type Ah[8];
    value_type Ao[8];
    // The the 8 neighbors is arranged as follow:
    // E:0, SE:1, S:2, SW:3, W:4, NW:5, N:6, NE:7
    // 5 6 7
    // 4 X 0
    // 3 2 1  
  };
#endif    


#ifndef PARAMETERS_DEPINAY_H
#define PARAMETERS_DEPINAY_H
  // Mosquito class includes general parameters/options for mosquito agents
  class ParameterClassDepinay
  {
  public:
    // ODE model parameters
    value_type rho25C;  // [1/hr]
    value_type EtpA;    // [cal/mol]
    value_type EtpL;    // [cal/mol]
    value_type EtpH;    // [cal/mol]
    value_type ThalfL;  // [K]
    value_type ThalfH;  // [K]  
  };
#endif  