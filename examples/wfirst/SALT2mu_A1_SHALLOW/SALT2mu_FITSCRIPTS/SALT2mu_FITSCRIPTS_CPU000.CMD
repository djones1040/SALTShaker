

# ------------------------------------
cd /project2/rkessler/SURVEYS/WFIRST/ROOT/SALT3/examples/wfirst/SALT2mu_A1_SHALLOW/ALLSurveys_Spec_DMS_SALT2
SALT2mu.exe pipeline/WFIRST_BBC.input  datafile=FITOPT000.FITRES prefix=SALT2mu_FITOPT000_MUOPT000   \
    >& SALT2mu_FITOPT000_MUOPT000.LOG 
mv SALT2mu_FITOPT000_MUOPT000.fitres SALT2mu_FITOPT000_MUOPT000.FITRES

# ------------------------------------
cd /project2/rkessler/SURVEYS/WFIRST/ROOT/SALT3/examples/wfirst/SALT2mu_A1_SHALLOW/AllSurveys_Spec_DMS_BYO
SALT2mu.exe pipeline/WFIRST_BBC.input  datafile=FITOPT000.FITRES prefix=SALT2mu_FITOPT000_MUOPT000   \
    >& SALT2mu_FITOPT000_MUOPT000.LOG 
mv SALT2mu_FITOPT000_MUOPT000.fitres SALT2mu_FITOPT000_MUOPT000.FITRES

 touch /project2/rkessler/SURVEYS/WFIRST/ROOT/SALT3/examples/wfirst/SALT2mu_A1_SHALLOW/SALT2mu_FITSCRIPTS/SALT2mu_FITSCRIPTS_CPU000.DONE 
