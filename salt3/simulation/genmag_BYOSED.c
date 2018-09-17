/*****************************************
  Created Sep 2018
  BYOSED = Build Your Own SED

  Lots of options to mangle and tweak an inital SED sequence
  such as Hsiao. Example options are to add random stretch,
  apply color law, add spectral features, correlate with
  host properties, etc ...

  Initial motivation is to build underlying "true" SED model to
  test SNIa model training. However, this function could in 
  principle be used for SNCC or other transients.

 *****************************************/

#include  <stdio.h> 
#include  <math.h>     
#include  <stdlib.h>   
#include  <sys/stat.h>

#include  "sntools.h"           // SNANA community tools
#include  "genmag_SEDtools.h"
#include  "genmag_SIMSED.h"
#include  "MWgaldust.h"


// =========================================================
void init_genmag_BYOSED(char *PATH_VERSION) {

  // Inputs:
  //  PATH_VERSION : points to model-param directory

  
  char fnam[] = "init_genmag_BYOSED" ;

  // -------------- BEGIN ------------------

  printf("\n xxx %s: path = '%s' \n", fnam, PATH_VERSION);

  // Read SED.INFO file for parameters characterizing how
  // to generate distributions; e.g., Asymmetric Gaussians,
  // top-hat, etc ...
  //
  // Need mechanism to pass command-line override parameters

  return ;

} // end init_BYOSED

// =========================================================
void genmag_BYOSED(double zCMB, double MU, double MWEBV, 
		   int IFILT, int NOBS, double *TOBS_list, 
		   double *MAGOBS_list, double *MAGERR_list ) {

  // Created Sep 2018
  //
  // Inputs:
  //   zCMB      : cmb redshift
  //   MU        : distance modulus
  //   MWEBV     : E(B-V) for Milky Wat
  //   IFILT     : filter index
  //   NOBS      : number of observations
  //   TOBS_list : list of MJD-PEAKMJD
  //
  // Outputs"
  //   MAGOBS_list   : list of true mags
  //   MAGERR_list   : list of mag errors (place-holder, in case)
  //

  int o;
  char fnam[] = "genmag_BYOSED" ;

  // ------------ BEGIN -----------

  for(o=0; o < NOBS; o++ ) { MAGOBS_list[o] = 99.0 ; }

  printf(" xxx %s: process z=%.3f MU=%.3f IFILT=%d \n",
	 fnam, zCMB, MU, IFILT);


  // Modules below could be Python or C

  // Generate random parameter values from distributions defined
  // in the SED.INFO file. 
  // E.g., color, stretch, amplitudes for spectral features, etc ...
  //    getPar_BYOSED();
  
  // Pass these parameter to module which creates SED
  //    getSED_BYOSED(...)

  // Pass SED to do integral and get mags
  // If this is Python, will need additional init functions
  // to pass filter trans and primary refs.
  //    getMag_BYOSED(...)



  return ;


} // end genmag_BYOSED


// ============================================================
//  FETCH UTILITIES TO RETURN EXTRA INFO TO MAIN PROGRAM
// =============================================================


// ================================================
int fetchParNames_BYOSED(char **parNameList) {

  // Pass name of each parameter to calling function, so that
  // these parameters can be included in the data files.
  // Function Returns number of parameters used to create
  // each SED.  **parNameList is a list of parameter names.
  //
  // Called once during init stage.

  int NPAR=0;
  char fnam[] = "fetchParNames_BYOSED" ;

  return(NPAR) ;

} // fetchParNames_BYOSED


void fetchParVal_BYOSED(double *parVal) {

  // return list of parameters to calling function (sim)
  // so that these parameters can be included in the
  // data files.
  //
  // Called once per event.
  
  int NPAR=0;
  char fnam[] = "fetchParVal_BYOSED" ;

  // ------------- BEGIN ------------------

  return ;

} // end fetchParVal_BYOSED

// =================================================
void fetchSED_BYOSED(double Tobs, int *NLAM, double *LAM, double *FLUX) {

  // return SED to calling function; e.g., to write spectra to data files.

  char fnam[] = "fetchSED_BYOSED" ;

  return;

} // end fetchSED_BYOSED
