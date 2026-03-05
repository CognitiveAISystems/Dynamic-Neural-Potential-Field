#include <l4casadi.hpp>

L4CasADi l4casadi("/app/NPField/script_d3_multi/_l4c_generated", "y_expr", 3, 1, 1, 10, "cuda", true, true, true, false, true, false);

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

// Function y_expr

static const casadi_int y_expr_s_in0[3] = { 3, 1, 1};
static const casadi_int y_expr_s_out0[3] = { 1, 10, 1};

// Only single input, single output is supported at the moment
CASADI_SYMBOL_EXPORT casadi_int y_expr_n_in(void) { return 1;}
CASADI_SYMBOL_EXPORT casadi_int y_expr_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT const casadi_int* y_expr_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return y_expr_s_in0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* y_expr_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return y_expr_s_out0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int y_expr(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.forward(arg[0], res[0]);
  return 0;
}

// Jacobian y_expr

CASADI_SYMBOL_EXPORT casadi_int jac_y_expr_n_in(void) { return 2;}
CASADI_SYMBOL_EXPORT casadi_int jac_y_expr_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT int jac_y_expr(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  l4casadi.jac(arg[0], res[0]);
  return 0;
}



// adj1 y_expr

CASADI_SYMBOL_EXPORT casadi_int adj1_y_expr_n_in(void) { return 3;}
CASADI_SYMBOL_EXPORT casadi_int adj1_y_expr_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT int adj1_y_expr(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  // adj1 [i0, out_o0, adj_o0] -> [out_adj_i0]
  l4casadi.adj1(arg[0], arg[2], res[0]);
  return 0;
}


// jac_adj1 y_expr

CASADI_SYMBOL_EXPORT casadi_int jac_adj1_y_expr_n_in(void) { return 4;}
CASADI_SYMBOL_EXPORT casadi_int jac_adj1_y_expr_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT int jac_adj1_y_expr(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  // jac_adj1 [i0, out_o0, adj_o0, out_adj_i0] -> [jac_adj_i0_i0, jac_adj_i0_out_o0, jac_adj_i0_adj_o0]
  if (res[1] != NULL) {
    l4casadi.invalid_argument("jac_adj_i0_out_o0 is not provided by L4CasADi. If you need this feature, please contact the L4CasADi developer.");
  }
  if (res[0] != NULL) {
    l4casadi.jac_adj1_p(arg[0], arg[2], res[0]);
  }
  if (res[2] != NULL) {
    l4casadi.jac_adj1_t(arg[0], arg[2], res[2]);
  }

  return 0;
}




#ifdef __cplusplus
} /* extern "C" */
#endif