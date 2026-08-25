/*******************************************************************************\

	def_settings.cpp in Sequential Minimal Optimization ver2.0
	
	implements initialization function for def_settings.
	
	Chu Wei Copyright(C) National Univeristy of Singapore
	Create on Jan. 16 2000 at Control Lab of Mechanical Engineering 
	Update on Aug. 23 2001 

\*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#ifndef __MACH__
    #include <malloc.h>
#endif
#include <string.h>
#include <math.h>
#include <time.h>
#include "smo.h"

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

/*******************************************************************************\

	def_Settings * Create_def_Settings ( char * filename ) 
	
	create and initialize the def_Settings structure 
	input:  the pointer to the name of Input Data File
	output: the pointer to the def_Settings structure

\*******************************************************************************/

def_Settings * Create_def_Settings ( char * filename ) 
{

	def_Settings * settings = NULL ;
	char * pstr = NULL ;
	unsigned int result = 0 ;
	char buf[LENGTH] = "" ;

	if (NULL == filename)
	{
		printf("\r\nFATAL ERROR : the input pointer is NULL.\r\n") ;
		return NULL ;
	}
	if (NULL == (settings = (def_Settings *)(malloc(sizeof(def_Settings)))))
	{
		printf("\r\nFATAL ERROR : fail to malloc def_settings.\r\n") ;
		return NULL ;
	}

	// set all elements as default values
	INPUTFILE = NULL ;
	TESTFILE = NULL ;
	VC = DEF_VC ;
	BETA = DEF_BETA ;
	EPSILON = DEF_EPSILON ;
	TOL = DEF_TOL ;
	KAPPA = DEF_KAPPA ;
	EPS = DEF_EPS ;
	P = DEF_P ;
	KERNEL = DEF_KERNEL ;
	SMO_DISPLAY = DEF_DISPLAY ;

	settings->ardon = DEF_ARDON ;

	settings->normalized_input = DEF_NORMALIZEINPUT ;
	settings->normalized_output = DEF_NORMALIZETARGET ;
	
	// save data file name in INPUTFILE
	if ( 0!=strlen(filename) && '-'!=filename[0] ) 
	{
		if ( NULL != INPUTFILE )
		{
			free( (void*) INPUTFILE ) ;
			INPUTFILE = NULL ;						
		}		
		if ( NULL == ( INPUTFILE = strdup(filename) ) )
		{
			// clear the structure before exit
			free (settings) ;
			printf("\r\nFATAL ERROR : fail to save the name of input file.\r\n") ;
			return NULL ;
		}
	}
	else
	{
		free (settings) ;
		return NULL ;
	}

	// if there is "train" in the file name of training data, such as "*train*.*",
	// test data set should be named as "*test*.*".
	// if we fail to find "train" in the training data file,
	// we just use the train data as test data.

	// create testing file name 
	pstr = strstr( INPUTFILE, "train" ) ;
	if (NULL == pstr)
		TESTFILE = strdup(filename) ;
	else
	{
		result = abs( INPUTFILE - pstr ) ;
		strncpy (buf, INPUTFILE, result ) ;
		buf[result] = '\0' ;
		strcat(buf, "test") ;
		strcat (buf, pstr+5) ;
		TESTFILE = strdup(buf) ;
	}
	
	Create_Data_List( &(settings->pairs) ) ;
	Create_Data_List( &(settings->testdata) ) ;
	Create_Data_List( &(settings->training) ) ;

	return settings ; 
}

def_Settings * Create_def_Settings_Python ( void )
{

	def_Settings * settings = NULL ;
	
	if (NULL == (settings = (def_Settings *)(malloc(sizeof(def_Settings)))))
	{
		printf("\r\nFATAL ERROR : fail to malloc def_settings.\r\n") ;
		return NULL ;
	}

	// set all elements as default values
	INPUTFILE = NULL ;
	TESTFILE = NULL ;
	VC = DEF_VC ;
	BETA = DEF_BETA ;
	EPSILON = DEF_EPSILON ;
	TOL = DEF_TOL ;
	KAPPA = DEF_KAPPA ;
	EPS = DEF_EPS ;
	P = DEF_P ;
	KERNEL = DEF_KERNEL ;
	SMO_DISPLAY = DEF_DISPLAY ;

	settings->ardon = DEF_ARDON ;

	settings->normalized_input = DEF_NORMALIZEINPUT ;
	settings->normalized_output = DEF_NORMALIZETARGET ;
	
	Create_Data_List( &(settings->pairs) ) ;
	Create_Data_List( &(settings->testdata) ) ;
	Create_Data_List( &(settings->training) ) ;

	return settings ; 
}

BOOL Update_def_Settings( def_Settings * defsetting )
{
	char buf[LENGTH] ;
	char msg[20] ;
	BOOL batchgoon = TRUE ;
	char * pstr ;
	unsigned int sz = 0 ;	
	unsigned int serialno = 0 ;
	
	if (NULL == defsetting)
		return FALSE ;
	if (defsetting->beta > 1.0 )
		defsetting->beta = 1.0 ;
	if ( FALSE == Is_Data_Empty(&defsetting->pairs) )
	{
		// create next input file name	
		pstr = strrchr( defsetting->inputfile, '.') ;	// 46
		if ( NULL != pstr )
			sz = abs( pstr - defsetting->inputfile ) ;		
		else
		{
			sz = strlen(defsetting->inputfile) ;
			return FALSE ;
		}
		strcpy (buf, pstr+1) ;
		serialno = atoi(buf) ;
		serialno += 1 ;
		strncpy( buf, defsetting->inputfile , sz ) ;
		buf[sz] = '\0' ;
		strcat( buf, "." ) ;
		// add appendix
		sprintf(msg,"%d",serialno) ;
		strcat( buf, msg ) ;
		// save inputfile name in def_Settings
		if ( NULL != defsetting->inputfile )
		{
			free( (void*) defsetting->inputfile ) ;
			defsetting->inputfile = NULL ;						
		}		
		if ( NULL == ( defsetting->inputfile = strdup(buf) ) )
			batchgoon = FALSE ;

		// create next test file name
		pstr = strrchr( defsetting->testfile, '.') ;	// 46
		if ( NULL != pstr )
			sz = abs( pstr - defsetting->testfile ) ;		
		else
			sz = strlen(defsetting->testfile) ;
		strcpy (buf, pstr+1) ;
		serialno = atoi(buf) ;
		serialno += 1 ;
		strncpy( buf, defsetting->testfile , sz ) ;
		buf[sz] = '\0' ;
		strcat( buf, "." ) ;
		// add appendix		
		sprintf(msg,"%d",serialno) ;
		strcat( buf, msg ) ;
		// save testfile name in def_Settings
		if ( NULL != defsetting->testfile )
		{
			free( (void*) defsetting->testfile ) ;
			defsetting->testfile = NULL ;						
		}		
		if ( NULL == ( defsetting->testfile = strdup(buf) ) )
			batchgoon = FALSE ;
		// load data into pairs 
		if ( FALSE == smo_Loadfile(&(defsetting->pairs), defsetting->inputfile, 0) )
			batchgoon = FALSE ;
	}
	else
	{
		// load data into pairs 
		if ( FALSE == smo_Loadfile(&(defsetting->pairs), defsetting->inputfile, 0) )
		{	
			batchgoon = FALSE ;
			printf("Failed to load training data from the file %s\n", defsetting->inputfile) ;
		}
#ifdef _PROSTATE_VIVO
		else if (defsetting->pairs.count!=100||defsetting->pairs.dimen!=6117||CLASSIFICATION!=defsetting->pairs.datatype)
		{	
			batchgoon = FALSE ;
			printf("The file %s is not ProstateABreast.VIVO.\n", defsetting->inputfile) ;
		}
#endif
	}	
	if ( CLASSIFICATION == defsetting->pairs.datatype )
	{	
		defsetting->beta = 1.0 ;
	}

	// load data into pairs 
	// load testing data into pairs
	//if (batchgoon == TRUE)
	//	if ( FALSE == smo_Loadfile(&(defsetting->testdata), defsetting->testfile, defsetting->pairs.dimen) )
	//		printf ("Failed to load testing data from the file %s", defsetting->testfile ) ;	
	
	return batchgoon ;	
}

/*******************************************************************************\

	void * Clear_def_Settings ( def_Settings * ptrSetting ) 
	
	Clear the def_Settings structure, including the unique copy of Data_List
	input:  the pointer to the def_Settings structure
	output:  none 

\*******************************************************************************/

void Clear_def_Settings( def_Settings * settings )
{
	if ( NULL != settings )
	{		
		if ( FALSE == Clear_Data_List( &(settings->pairs) ) )
			printf("\r\nFATAL ERROR : error happened in Clearing Data_List.\r\n") ;
		if ( FALSE == Clear_Data_List( &(settings->testdata) ) )
			printf("\r\nFATAL ERROR : error happened in Clearing Data_List.\r\n") ;
		if (NULL!=settings->training.featuretype)
				free(settings->training.featuretype) ;
		settings->training.featuretype = NULL ;
		if (NULL!=settings->training.labelnum)
				free(settings->training.labelnum) ;
		settings->training.labelnum = NULL ;
		if (NULL!=settings->training.labels)
				free(settings->training.labels) ;
		settings->training.labels = NULL ;

		if ( NULL != INPUTFILE )
			free( INPUTFILE ) ;	
		if ( NULL != TESTFILE )
			free( TESTFILE ) ;	
		free (settings) ;
		settings = NULL ;
	}
}



 

/*
// end of def_settings.c */
