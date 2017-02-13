#include"Storage_Structure.h"
#include"Global_Config.h"
#include<mkl.h>
#include<iostream>
//Init_Ftr_Mtx_Strc Modify with IRMF
void ftr_mtx_strc::Init_Ftr_Mtx_Strc(){
	//normal distribution
	vslNewStream(&stream, VSL_BRNG_MCG31, 1);
	for (int i = 0; i < U_NUM; i++)
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, MATRIX_RANK, usr_mtx[0][i], 0.0, SEG / 2);
	for (int i = 0; i < I_NUM; i++)
		vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, MATRIX_RANK, itm_mtx[0][i], 0.0, SEG / 2);
	//uniform distribution
	/*for (int i = 0; i < U_NUM; i++)
	for (int j = 0; j < MATRIX_RANK; j++)
		usr_mtx[0][i][j] = (rand() / double(RAND_MAX) * 2 * SEG - SEG);
	for (int i = 0; i < I_NUM; i++)
	for (int j = 0; j < MATRIX_RANK; j++)
		itm_mtx[0][i][j] = (rand() / double(RAND_MAX) * 2 * SEG - SEG);*/
	return;
}