#ifndef GLOBAL_FUNCTION
#define GLOBAL_FUNCTION
#include "Storage_Structure.h"
int Glb_Init(ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss);
int Glb_Inpt_Init(int data_idx, ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss);

int Batch_Train(ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss);

int mdf_PtoQ(int cur_epch, int usr_idx, double* det_Pu, ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss);
int mdf_QtoP(int cur_epch, int itm_idx, double* det_Qh, ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss);
int mdf_cmb(int usr_idx, int itm_idx, double cur_rat, ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss);

int Incre_Train(int rat_num,ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss);
#endif