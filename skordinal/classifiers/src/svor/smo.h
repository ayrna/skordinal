/*******************************************************************************\
	smo.h in Sequential Minimal Optimization ver2.0
	UNIFIED SVOREX & SVORIM
\*******************************************************************************/
#ifdef  __cplusplus
extern "C" {
#endif

#ifndef _SMO_H
#define _SMO_H

#ifdef _WIN_SIMU
#include <windows.h>
#else
typedef enum _BOOL { FALSE = 0, TRUE = 1 } BOOL ;
#define min(a,b)        ((a) < (b) ? (a) : (b))
#define max(a,b)        ((a) > (b) ? (a) : (b))
#endif

#define MINNUM          (2)			
#define LENGTH          (307200)		 

typedef enum _Set_Name { Io_a=5, Io_b=6, I_One=1, I_Two=2, I_Fou=4, I_Thr=3, I_o=0 } Set_Name ;  
typedef enum _Data_Type { REGRESSION=2, CLASSIFICATION=1, ORDINAL=3, UNKNOWN=0 } Data_Type ;
typedef enum _Method_Name { SMO_SKONE, SMO_SKTWO } Method_Name ;
typedef enum _Kernel_Name { GAUSSIAN=0, POLYNOMIAL=1, LINEAR=2 } Kernel_Name ;
typedef enum _Training_Method { BAYESIAN=1, SMOMERELY=2, CROSSVALIDATION=3 } Training_Method ;

typedef struct _Data_Node {
	long unsigned int index;
	unsigned int count;             
	int fold;
	double * point;                
	unsigned int target;                 
	double guess;					
	double fx;
	struct _Data_Node * next;      
} Data_Node ;

typedef struct _Data_List {
	Data_Type datatype;            
	BOOL normalized_input;			 
	BOOL normalized_output;		
	unsigned long int count;       
	unsigned long int dimen;            	
	unsigned int i_ymax;
	unsigned int i_ymin;
	unsigned int classes;
	char * filename;
	unsigned int * labels;
	unsigned int * labelnum;
	double mean;                   
	double deviation;              	
	int * featuretype;					
	double * x_mean;				
	double * x_devi;				
	Data_Node * front;          
	Data_Node * rear;             
} Data_List ;

typedef struct _Cache_Node {
	double new_Fi;
	struct _Alphas * alpha;
	struct _Cache_Node * previous;
	struct _Cache_Node * next;
} Cache_Node ;

typedef struct _Cache_List {
	long unsigned int count;
	Cache_Node * front;
	Cache_Node * rear;
} Cache_List ;

typedef struct _Alphas {
	double alpha;              // Shared scalar output for Prediction
	double alpha_up;           // SVOREX
	double alpha_dw;           // SVOREX
	Set_Name setname_up;       // SVOREX
	Set_Name setname_dw;       // SVOREX
    
	double * alpha_ptr;        // SVORIM
	Set_Name * setname_ptr;    // SVORIM

	double f_cache;					 
	double * kernel;					
	Data_Node * pair;					
	Cache_Node * cache;				
} Alphas ;

typedef struct _smo_Settings {
	int model_type; // 0: SVOREX, 1: SVORIM
	double vc ;                     /*/ Regularization Parameter*/
	
	/*/ introduce for imbalanced datasets*/
	double vc_p ;                   /*/ C for positive labelled samples*/
	double vc_n ;                   /*/ C for negative labelled samples*/

	double epsilon ;                /*/ Epsilon insensitive Loss Function*/
	double beta ;					/*/ Soft Insensitive Loss Function	*/
	double tol ;                    /*/ Tolerance Parameter in Loose KKT */
	double eps ;					/*/ Error Precision Setting*/
	double duration ;               /*/ clock time passed*/
	double * ard ;

	Kernel_Name kernel;            
	unsigned int p;               
	double kappa;					

	struct _Alphas * alpha;		
	struct _Cache_List io_cache;	
	struct _Data_List * pairs;		
	Method_Name method;        

	long unsigned int * ij_low;      
	long unsigned int * ij_up;      
	double * bj_low;               
	double * bj_up;                 
	double * biasj;
	double * mu;                    // SVOREX
	double * bmu_low;               // SVOREX
	double * bmu_up;                // SVOREX
	long unsigned int * imu_low;    // SVOREX
	long unsigned int * imu_up;     // SVOREX
	double bias;
	
	BOOL smo_display;			
	BOOL smo_working;			
	double smo_timing;			
	char * inputfile;			
	char * dumpingfile;		
	unsigned long int cache_size;  
	BOOL cacheall;
	BOOL ardon;
	
	double testerror;
	double testrate;
	double c1p, c2p, c1n, c2n, svs;
	int index;			
	BOOL abort;		
	BOOL smo_balance;	
} smo_Settings ;

typedef struct _def_Settings {
	int model_type; // 0: SVOREX, 1: SVORIM
	double vc;
	double vc_p ;                   /*/ Regularization Parameter*/
	double vc_n ;                   /*/ Regularization Parameter*/                  
	double tol;                    
	double eps;                   
	double beta;
	double epsilon;
	Kernel_Name kernel;            
	double kappa;                  
	unsigned int p;                
	Method_Name method;            
	BOOL smo_display;
	BOOL smo_balance;
	BOOL ardon;

	unsigned int index, loops, seeds, kfold, repeat;
	unsigned long int cache_size;  

	double lnC_start, lnC_end, lnC_step;
	double lnK_start, lnK_end, lnK_step;	
	double best_rate;
	double def_lnC_start, def_lnC_end, def_lnC_step;
	double def_lnK_start, def_lnK_end, def_lnK_step;
	double zoomin, time;
	
	char * inputfile;              
	char * testfile;               
	struct _Data_List pairs;		
	struct _Data_List training;
	struct _Data_List validation;
	struct _Data_List testdata;		
	
	BOOL normalized_input;			
	BOOL normalized_output;		
	Training_Method trainmethod;	
} def_Settings ;

#define SMO_WORKING    (settings->smo_working) 
#define SMO_DISPLAY    (settings->smo_display) 
#define EPS            (settings->eps) 
#define TOL            (settings->tol) 
#define VC             (settings->vc) 
#define KAPPA          (settings->kappa) 
#define P              (settings->p)
#define METHOD         (settings->method) 
#define KERNEL         (settings->kernel) 
#define BIAS		   (settings->bias)
#define INDEX          (settings->index)
#define DURATION       (settings->duration) 
#define Io_CACHE       (settings->io_cache) 
#define ALPHA          (settings->alpha)
#define INPUTFILE      (settings->inputfile) 
#define TESTFILE       (settings->testfile) 
#define EPSILON        (settings->epsilon)
#define KFOLD          (settings->kfold) 
#define BETA           (settings->beta) 

/*/ default settings*/
#define DEF_EPS          (0.000001)
#define DEF_TOL          (0.001) 
#define DEF_EPSILON      (0.1)
#define DEF_BETA         (0) 
#define DEF_VC           (1.0)
#define DEF_KAPPA        (1.0) 
#define DEF_P            (1) 
#define DEF_KERNEL       (GAUSSIAN)
#define DEF_METHOD       (SMO_SKTWO) 
#define DEF_DISPLAY      (FALSE)
#define DEF_ARDON		 (FALSE)
#define DEF_NORMALIZEINPUT    (FALSE)
#define DEF_NORMALIZETARGET   (FALSE)
#define DEF_SUPERLNC		  (2)	
#define DEF_INFERLNC		  (-1) 
#define DEF_SUPERLNK     (1)
#define DEF_INFERLNK     (-2)
#define DEF_TRAINING	 (CROSSVALIDATION)
#define DEF_KFOLD	     (5)
#define DEF_COARSESTEP   (0.5)
#define DEF_REFINESTEP   (0.1)
#define DEF_CACHE        (5000)
#define DEF_ZOOMIN       (5)
#define DEF_REPEAT       (1) 
#define DEF_LOOP         (2)
#define DEF_BALANCE      (FALSE)

def_Settings * Create_def_Settings_Python ( void );
void Clear_def_Settings( def_Settings * settings ) ;

BOOL Create_Data_List ( Data_List * list ) ;
BOOL Is_Data_Empty ( Data_List * list ) ;
BOOL Clear_Data_List ( Data_List * list ) ;
BOOL Add_Data_List ( Data_List * list, Data_Node * node ) ;
Data_Node * Create_Data_Node ( long unsigned int index, double * point, unsigned int y ) ;
BOOL Clear_Label_Data_List ( Data_List * list ) ;
int Add_Label_Data_List ( Data_List * list, Data_Node * node ) ;

BOOL smo_Loadfile ( Data_List * pairs, char * inputfilename, int inputdim );

smo_Settings * Create_smo_Settings_Python ( def_Settings * settings ) ;
void Clear_smo_Settings( smo_Settings * settings ) ;

BOOL Create_Cache_List( Cache_List * ) ;
BOOL Clear_Cache_List( Cache_List * ) ;
BOOL Is_Cache_Empty( Cache_List * ) ;
BOOL Add_Cache_Node( Cache_List *, Alphas * ) ;
BOOL Sort_Cache_Node( Cache_List *, Alphas * ) ;
BOOL Del_Cache_Node( Cache_List *, Alphas * ) ; 

Alphas * Create_Alphas( smo_Settings * ) ;
BOOL Clean_Alphas ( Alphas *, smo_Settings * ) ;
BOOL Check_Alphas ( Alphas *, smo_Settings * ) ;
BOOL Clear_Alphas ( smo_Settings * ) ;

double Calc_Kernel( Alphas * , Alphas * , smo_Settings * ) ;
double Calculate_Kernel( double * , double * , smo_Settings * ) ;

double Calculate_Ordinal_Fi( long unsigned int i, smo_Settings * settings ) ;
Set_Name Get_Ordinal_Label( Alphas * , unsigned int, smo_Settings * settings) ;
Set_Name Get_UP_Label ( Alphas * alpha, smo_Settings * settings) ;
Set_Name Get_DW_Label ( Alphas * alpha, smo_Settings * settings) ;
BOOL Is_Io( Alphas * alpha, smo_Settings * settings ) ;

BOOL smo_routine_Python ( smo_Settings * settings ) ;
BOOL ordinal_examine_example ( Alphas * alpha, smo_Settings * settings ) ;
unsigned int active_cross_threshold (smo_Settings * settings) ;
unsigned int active_threshold (smo_Settings * settings) ;

BOOL ordinal_takestep ( Alphas * alpha1, Alphas * alpha2, unsigned int threshold, smo_Settings * settings ) ;
BOOL ordinal_cross_takestep ( Alphas * alpha4, unsigned int, Alphas * alpha5, unsigned int, smo_Settings * settings ) ;
BOOL ordinal_cross_identical ( Alphas * alpha1, Alphas * alpha2, unsigned int threshold, smo_Settings * settings ) ;
BOOL Decide_Boundary (double gamma, int s1, int s2, smo_Settings * settings, double * H, double * L);

BOOL svm_predict(Data_List * test, smo_Settings * settings);
BOOL svm_predict_Python(Data_List * test, smo_Settings * settings);

void tstart(void) ;
void tend(void) ;
double tval() ;

#endif
#ifdef  __cplusplus
}
#endif