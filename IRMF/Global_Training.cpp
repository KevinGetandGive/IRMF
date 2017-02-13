#include"Storage_Structure.h"
#include"Global_Config.h"
#include<mkl.h>
#include<cmath>
#include<cstdio>
#include<omp.h>
#include<string>
#include<cstring>
int Batch_Train(ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss){
	for (int i = 1; i <= ITERATION_NUM; i++){
		for (int usr_idx = 0; usr_idx < U_NUM; usr_idx++){
			if (cur_rms.lega_usr[usr_idx]== false)
				continue;
			double rela_num = 0;
			cblas_dscal(MATRIX_RANK, 0.0, &cur_fms.usr_mtx[i][usr_idx][0], 1);
			for (int k = cur_rms.lnk_head[usr_idx]; k != 0; k =cur_rms.lnk_node[k].nex_elem){
				int itm_idx = cur_rms.lnk_node[k].elem_info - U_NUM;
				double err = cur_rms.lnk_node[k].elem_rat - cblas_ddot(MATRIX_RANK, &cur_fms.usr_mtx[i - 1][usr_idx][0], 1, &cur_fms.itm_mtx[i - 1][itm_idx][0], 1);
				cblas_daxpy(MATRIX_RANK, STEP_SIZE*err, &cur_fms.itm_mtx[i - 1][itm_idx][0], 1, &cur_fms.usr_mtx[i][usr_idx][0], 1);
				rela_num += 1.0;
			}
			double tmp_pow = pow(1 - STEP_SIZE*REGULARIZED_LAMBDA, rela_num);
			double alpha_k = (1 - tmp_pow) / ((STEP_SIZE*REGULARIZED_LAMBDA)*rela_num);
			cblas_dscal(MATRIX_RANK, alpha_k,&cur_fms.usr_mtx[i][usr_idx][0], 1);
			cblas_daxpy(MATRIX_RANK, tmp_pow, &cur_fms.usr_mtx[i - 1][usr_idx][0], 1, &cur_fms.usr_mtx[i][usr_idx][0], 1);
		}
		for (int j = 0; j < I_NUM; j++){
			if (cur_rms.lega_itm[j]==false)
				continue;
			int itm_idx = j;
			double rela_num = 0;
			cblas_dscal(MATRIX_RANK, 0.0,&cur_fms.itm_mtx[i][itm_idx][0], 1);
			for (int k =cur_rms.lnk_head[itm_idx+U_NUM]; k != 0; k = cur_rms.lnk_node[k].nex_elem){
				int usr_idx = cur_rms.lnk_node[k].elem_info;
				double err = cur_rms.lnk_node[k].elem_rat - cblas_ddot(MATRIX_RANK, &cur_fms.usr_mtx[i - 1][usr_idx][0], 1, &cur_fms.itm_mtx[i - 1][itm_idx][0], 1);
				cblas_daxpy(MATRIX_RANK, STEP_SIZE*err, &cur_fms.usr_mtx[i - 1][usr_idx][0], 1, &cur_fms.itm_mtx[i][itm_idx][0], 1);
				rela_num += 1.0;
			}
			double tmp_pow = pow(1 - STEP_SIZE*REGULARIZED_LAMBDA, rela_num);
			double beta_h = (1 - tmp_pow) / ((STEP_SIZE*REGULARIZED_LAMBDA)*rela_num);
			cblas_dscal(MATRIX_RANK, beta_h, &cur_fms.itm_mtx[i][itm_idx][0], 1);
			cblas_daxpy(MATRIX_RANK, tmp_pow, &cur_fms.itm_mtx[i - 1][itm_idx][0], 1, &cur_fms.itm_mtx[i][itm_idx][0], 1);
		}
	}
	return 0;
}


int mdf_PtoQ(int cur_epch, int usr_idx, double* det_Pu,ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss){
    int sam_num=0;
	double* tmp_fea;
	tmp_fea = (double*)malloc(sizeof(double)*MATRIX_RANK);
	for (int i = cur_rms.lnk_head[usr_idx]; i != 0; i = cur_rms.lnk_node[i].nex_elem){
        if (sam_used[(i + 1) / 2] == false)
            sam_num++;
        sam_used[(i + 1) / 2] = true;
		int itm_idx = cur_rms.lnk_node[i].elem_info - U_NUM;
		cblas_dcopy(MATRIX_RANK, det_Pu, 1, tmp_fea, 1);
		double err = cur_rms.lnk_node[i].elem_rat - cblas_ddot(MATRIX_RANK,&cur_fms.usr_mtx[cur_epch][usr_idx][0], 1,&cur_fms.itm_mtx[cur_epch][itm_idx][0], 1);// u_f[now_epoch][uid] had been modified
		cblas_dscal(MATRIX_RANK, err, tmp_fea, 1);
		double ano_err = cblas_ddot(MATRIX_RANK, det_Pu, 1, &cur_fms.itm_mtx[cur_epch][itm_idx][0],1);
		cblas_daxpy(MATRIX_RANK, -1.0*ano_err,&cur_fms.usr_mtx[cur_epch][usr_idx][0], 1, tmp_fea, 1);
		cblas_daxpy(MATRIX_RANK, ano_err, det_Pu, 1, tmp_fea, 1);
		double beta_hk = (1 - pow(1 - STEP_SIZE*REGULARIZED_LAMBDA, cur_rms.itm_num[itm_idx])) / ((STEP_SIZE*REGULARIZED_LAMBDA)*cur_rms.itm_num[itm_idx]);
		cblas_dscal(MATRIX_RANK, beta_hk*STEP_SIZE, tmp_fea, 1);
		cblas_daxpy(MATRIX_RANK, 1.0, tmp_fea, 1, &cur_fms.itm_mtx[cur_epch+1][itm_idx][0], 1);
	}
	free(tmp_fea);
	return sam_num;
}

int mdf_QtoP(int cur_epch, int itm_idx, double* det_Qh, ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss){
    int sam_num=0;
	double* tmp_fea;
	tmp_fea = (double*)malloc(sizeof(double)*MATRIX_RANK);
	for (int i = cur_rms.lnk_head[itm_idx + U_NUM]; i != 0; i = cur_rms.lnk_node[i].nex_elem){
        if (sam_used[(i + 1) / 2] == false)
            sam_num++;
        sam_used[(i + 1) / 2] = true;
		int usr_idx = cur_rms.lnk_node[i].elem_info;
		cblas_dcopy(MATRIX_RANK, det_Qh, 1, tmp_fea, 1);
		double err = cur_rms.lnk_node[i].elem_rat - cblas_ddot(MATRIX_RANK,&cur_fms.itm_mtx[cur_epch][itm_idx][0], 1,&cur_fms.usr_mtx[cur_epch][usr_idx][0], 1);
		cblas_dscal(MATRIX_RANK, err, tmp_fea, 1);
		double ano_err = cblas_ddot(MATRIX_RANK, det_Qh, 1,&cur_fms.usr_mtx[cur_epch][usr_idx][0], 1);
		cblas_daxpy(MATRIX_RANK, -1.0*ano_err,&cur_fms.itm_mtx[cur_epch][itm_idx][0], 1, tmp_fea, 1);
		cblas_daxpy(MATRIX_RANK, ano_err, det_Qh, 1, tmp_fea, 1);
		double alpha_hk = (1 - pow(1 - STEP_SIZE*REGULARIZED_LAMBDA, cur_rms.usr_num[usr_idx])) / ((STEP_SIZE*REGULARIZED_LAMBDA)*cur_rms.usr_num[usr_idx]);
		cblas_dscal(MATRIX_RANK, alpha_hk*STEP_SIZE, tmp_fea, 1);
		cblas_daxpy(MATRIX_RANK, 1.0, tmp_fea, 1,&cur_fms.usr_mtx[cur_epch+1][usr_idx][0], 1);
	}
	free(tmp_fea);
	return sam_num;
}

int mdf_cmb(int usr_idx,int itm_idx,double cur_rat,ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss){
    int sam_num=0;
	double* pre_Pu;
	double* det_Pu;
	double* pre_Qh;
	double* det_Qh;
	pre_Pu = (double*)malloc(sizeof(double)*MATRIX_RANK);
	det_Pu = (double*)malloc(sizeof(double)*MATRIX_RANK);
	pre_Qh = (double*)malloc(sizeof(double)*MATRIX_RANK);
	det_Qh = (double*)malloc(sizeof(double)*MATRIX_RANK);
	
	//modify pu1
	cblas_dcopy(MATRIX_RANK,&cur_fms.usr_mtx[1][usr_idx][0], 1, pre_Pu, 1);
	double tmp_pow = pow(1 - STEP_SIZE*REGULARIZED_LAMBDA, cur_rms.usr_num[usr_idx]);
	double alpha_kp1 = (1 - tmp_pow*(1 - STEP_SIZE*REGULARIZED_LAMBDA)) / ((STEP_SIZE*REGULARIZED_LAMBDA)*(cur_rms.usr_num[usr_idx] + 1));
	double alpha_k;
	if (cur_rms.usr_num[usr_idx] == 0)
		alpha_k = 0;
	else
		alpha_k = (1 - tmp_pow) / ((STEP_SIZE*REGULARIZED_LAMBDA)*cur_rms.usr_num[usr_idx]);	

	if (cur_rms.usr_num[usr_idx] != 0){
		cblas_daxpy(MATRIX_RANK, -1.0*tmp_pow,&cur_fms.usr_mtx[0][usr_idx][0], 1, &cur_fms.usr_mtx[1][usr_idx][0], 1);
		cblas_dscal(MATRIX_RANK, alpha_kp1 / alpha_k,&cur_fms.usr_mtx[1][usr_idx][0], 1);
	}

	cblas_daxpy(MATRIX_RANK, tmp_pow*(1 - STEP_SIZE*REGULARIZED_LAMBDA),&cur_fms.usr_mtx[0][usr_idx][0], 1, &cur_fms.usr_mtx[1][usr_idx][0], 1);
	double err = cur_rat - cblas_ddot(MATRIX_RANK, &cur_fms.usr_mtx[0][usr_idx][0], 1,&cur_fms.itm_mtx[0][itm_idx][0], 1);
	cblas_daxpy(MATRIX_RANK, alpha_kp1*STEP_SIZE*err, &cur_fms.itm_mtx[0][itm_idx][0], 1,&cur_fms.usr_mtx[1][usr_idx][0], 1);
	cblas_dcopy(MATRIX_RANK, &cur_fms.usr_mtx[1][usr_idx][0], 1, det_Pu, 1);
	cblas_daxpy(MATRIX_RANK, -1.0, pre_Pu, 1, det_Pu, 1);

	//modify qh1
	cblas_dcopy(MATRIX_RANK,&cur_fms.itm_mtx[1][itm_idx][0], 1, pre_Qh, 1);
	tmp_pow = pow(1 - STEP_SIZE*REGULARIZED_LAMBDA, cur_rms.itm_num[itm_idx]);
	double beta_hp1 = (1 - tmp_pow*(1 - STEP_SIZE*REGULARIZED_LAMBDA)) / ((STEP_SIZE*REGULARIZED_LAMBDA)*(cur_rms.itm_num[itm_idx] + 1));
	double beta_h;
	if (cur_rms.itm_num[itm_idx] == 0)
		beta_h = 0;
	else
		beta_h = (1 - tmp_pow) / ((STEP_SIZE*REGULARIZED_LAMBDA)*cur_rms.itm_num[itm_idx]);

	//printf("BETA_H = %lf\n",beta_h);	

	if (cur_rms.itm_num[itm_idx] != 0){
		cblas_daxpy(MATRIX_RANK, -1.0*tmp_pow,&cur_fms.itm_mtx[0][itm_idx][0], 1,&cur_fms.itm_mtx[1][itm_idx][0], 1);
		cblas_dscal(MATRIX_RANK, beta_hp1 / beta_h, &cur_fms.itm_mtx[1][itm_idx][0], 1);
	}
	cblas_daxpy(MATRIX_RANK, tmp_pow*(1 - STEP_SIZE*REGULARIZED_LAMBDA), &cur_fms.itm_mtx[0][itm_idx][0], 1, &cur_fms.itm_mtx[1][itm_idx][0], 1);
	err = cur_rat - cblas_ddot(MATRIX_RANK, &cur_fms.usr_mtx[0][usr_idx][0], 1, &cur_fms.itm_mtx[0][itm_idx][0], 1);
	cblas_daxpy(MATRIX_RANK, beta_hp1*STEP_SIZE*err, &cur_fms.usr_mtx[0][usr_idx][0], 1, &cur_fms.itm_mtx[1][itm_idx][0], 1);
	cblas_dcopy(MATRIX_RANK, &cur_fms.itm_mtx[1][itm_idx][0], 1, det_Qh, 1);
	cblas_daxpy(MATRIX_RANK, -1.0, pre_Qh, 1, det_Qh, 1);

	//printf("%d,%d = %lf  &&  %d,%d = %lf\n",uid,u_cnt[uid],alpha_k,iid,i_cnt[iid],beta_h);
	//modify pud qhd
	for (int i = 2; i <= ITERATION_NUM; i++){
		sam_num+=mdf_PtoQ(i - 1, usr_idx, det_Pu,cur_fms,cur_rms,cur_ss);
		sam_num+=mdf_QtoP(i - 1, itm_idx, det_Qh,cur_fms,cur_rms,cur_ss);
		//pu modify 
		cblas_dcopy(MATRIX_RANK, &cur_fms.usr_mtx[i][usr_idx][0], 1, pre_Pu, 1);
		double tmp_pow = pow(1 - STEP_SIZE*REGULARIZED_LAMBDA, cur_rms.usr_num[usr_idx]);
		double alpha_kp1 = (1 - tmp_pow*(1 - STEP_SIZE*REGULARIZED_LAMBDA)) / ((STEP_SIZE*REGULARIZED_LAMBDA)*(cur_rms.usr_num[usr_idx] + 1));
		double alpha_k;
		if (cur_rms.usr_num[usr_idx] == 0)
			alpha_k = 0;
		else
			alpha_k = (1 - tmp_pow) / ((STEP_SIZE*REGULARIZED_LAMBDA)*cur_rms.usr_num[usr_idx]);

		if (cur_rms.usr_num[usr_idx] != 0){
			cblas_daxpy(MATRIX_RANK, -1.0*tmp_pow, &cur_fms.usr_mtx[i-1][usr_idx][0], 1, &cur_fms.usr_mtx[i][usr_idx][0], 1);
			cblas_daxpy(MATRIX_RANK, 1.0*tmp_pow, det_Pu, 1, &cur_fms.usr_mtx[i][usr_idx][0], 1);
			cblas_dscal(MATRIX_RANK, alpha_kp1 / alpha_k, &cur_fms.usr_mtx[i][usr_idx][0], 1);
		}

		cblas_daxpy(MATRIX_RANK, tmp_pow*(1 - STEP_SIZE*REGULARIZED_LAMBDA), &cur_fms.usr_mtx[i-1][usr_idx][0], 1, &cur_fms.usr_mtx[i][usr_idx][0], 1);
		double err = cur_rat - cblas_ddot(MATRIX_RANK, &cur_fms.usr_mtx[i-1][usr_idx][0], 1, &cur_fms.itm_mtx[i-1][itm_idx][0], 1) + cblas_ddot(MATRIX_RANK, &cur_fms.usr_mtx[i-1][usr_idx][0], 1, det_Qh, 1);
		cblas_daxpy(MATRIX_RANK, alpha_kp1*STEP_SIZE*err, &cur_fms.itm_mtx[i-1][itm_idx][0], 1, &cur_fms.usr_mtx[i][usr_idx][0], 1);
		cblas_daxpy(MATRIX_RANK, -1.0*alpha_kp1*STEP_SIZE*err, det_Qh, 1, &cur_fms.usr_mtx[i][usr_idx][0], 1);
		double tmp_dou = 0;
		for (int j = cur_rms.lnk_head[usr_idx]; j != 0; j = cur_rms.lnk_node[j].nex_elem){
            if (sam_used[(j + 1) / 2] == false)
                sam_num++;
            sam_used[(j + 1) / 2] = true;
			int tmp_itm_idx = cur_rms.lnk_node[j].elem_info - U_NUM;
			tmp_dou = cblas_ddot(MATRIX_RANK, det_Pu, 1, &cur_fms.itm_mtx[i-1][tmp_itm_idx][0], 1);
			cblas_daxpy(MATRIX_RANK, -1.0*alpha_kp1*STEP_SIZE*tmp_dou, &cur_fms.itm_mtx[i-1][tmp_itm_idx][0], 1, &cur_fms.usr_mtx[i][usr_idx][0], 1);
		}
		cblas_dcopy(MATRIX_RANK, &cur_fms.usr_mtx[i][usr_idx][0], 1, det_Pu, 1);
		cblas_daxpy(MATRIX_RANK, -1.0, pre_Pu, 1, det_Pu, 1);

		//qh modify
		cblas_dcopy(MATRIX_RANK, &cur_fms.itm_mtx[i][itm_idx][0], 1, pre_Qh, 1);
		tmp_pow = pow(1 - STEP_SIZE*REGULARIZED_LAMBDA, cur_rms.itm_num[itm_idx]);
		double beta_hp1 = (1 - tmp_pow*(1 - STEP_SIZE*REGULARIZED_LAMBDA)) / ((STEP_SIZE*REGULARIZED_LAMBDA)*(cur_rms.itm_num[itm_idx] + 1));
		double beta_h;
		if (cur_rms.itm_num[itm_idx] == 0)
			beta_h = 0;
		else
			beta_h = (1 - tmp_pow) / ((STEP_SIZE*REGULARIZED_LAMBDA)*cur_rms.itm_num[itm_idx]);

		if (cur_rms.itm_num[itm_idx] != 0){
			cblas_daxpy(MATRIX_RANK, -1.0*tmp_pow, &cur_fms.itm_mtx[i-1][itm_idx][0], 1, &cur_fms.itm_mtx[i][itm_idx][0], 1);
			cblas_daxpy(MATRIX_RANK, 1.0*tmp_pow, det_Qh, 1, &cur_fms.itm_mtx[i][itm_idx][0], 1);
			cblas_dscal(MATRIX_RANK, beta_hp1 / beta_h, &cur_fms.itm_mtx[i][itm_idx][0], 1);
		}
		cblas_daxpy(MATRIX_RANK, tmp_pow*(1 - STEP_SIZE*REGULARIZED_LAMBDA), &cur_fms.itm_mtx[i-1][itm_idx][0], 1, &cur_fms.itm_mtx[i][itm_idx][0], 1);
		err = cur_rat - cblas_ddot(MATRIX_RANK, &cur_fms.itm_mtx[i-1][itm_idx][0], 1, &cur_fms.usr_mtx[i-1][usr_idx][0], 1) + cblas_ddot(MATRIX_RANK, &cur_fms.itm_mtx[i-1][itm_idx][0], 1, det_Pu, 1);
		cblas_daxpy(MATRIX_RANK, beta_hp1*STEP_SIZE*err, &cur_fms.usr_mtx[i-1][usr_idx][0], 1, &cur_fms.itm_mtx[i][itm_idx][0], 1);
		cblas_daxpy(MATRIX_RANK, -1.0*beta_hp1*STEP_SIZE*err, det_Pu, 1, &cur_fms.itm_mtx[i][itm_idx][0], 1);
		tmp_dou = 0;
		for (int j = cur_rms.lnk_head[itm_idx+U_NUM]; j != 0; j = cur_rms.lnk_node[j].nex_elem){
            if (sam_used[(j + 1) / 2] == false)
                sam_num++;
            sam_used[(j + 1) / 2] = true;
			int tmp_usr_idx = cur_rms.lnk_node[j].elem_info;
			tmp_dou = cblas_ddot(MATRIX_RANK, det_Qh, 1, &cur_fms.usr_mtx[i-1][tmp_usr_idx][0], 1);
			cblas_daxpy(MATRIX_RANK, -1.0*beta_hp1*STEP_SIZE*tmp_dou, &cur_fms.usr_mtx[i-1][tmp_usr_idx][0], 1, &cur_fms.itm_mtx[i][itm_idx][0], 1);
		}
		cblas_dcopy(MATRIX_RANK, &cur_fms.itm_mtx[i][itm_idx][0], 1, det_Qh, 1);
		cblas_daxpy(MATRIX_RANK, -1.0, pre_Qh, 1, det_Qh, 1);
		double now_rat = cblas_ddot(MATRIX_RANK, cur_fms.usr_mtx[ITERATION_NUM][usr_idx], 1, cur_fms.itm_mtx[ITERATION_NUM][itm_idx], 1);
	}
	free(pre_Pu);
	free(det_Pu);
	free(pre_Qh);
	free(det_Qh);
	return sam_num;
}

int Incre_Train(int rat_num,ftr_mtx_strc& cur_fms, rat_mtx_strc& cur_rms, stttc_strc& cur_ss){
	int *last_pos;
    int sam_num=0;
    memset(sam_used,0,sizeof(sam_used));
	last_pos = (int*)malloc(sizeof(int*)*(U_NUM + I_NUM));
	memset(last_pos, -1, sizeof(int)*(U_NUM + I_NUM));
	int i = cur_rms.rela_num / 2;
	while (i < rat_num){
		last_pos[cur_rms.inpt_elem[i].usr_idx] = i;
		last_pos[cur_rms.inpt_elem[i].itm_idx+U_NUM] = i;
		i++;
	}
	i = cur_rms.rela_num / 2;
	while (i < rat_num){
		cur_rms.lega_usr[cur_rms.inpt_elem[i].usr_idx] = true;
		cur_rms.lega_itm[cur_rms.inpt_elem[i].itm_idx] = true;
		double tmp_resi = cur_rms.inpt_elem[i].rat - cblas_ddot(MATRIX_RANK, cur_fms.usr_mtx[ITERATION_NUM][cur_rms.inpt_elem[i].usr_idx], 1, cur_fms.itm_mtx[ITERATION_NUM][cur_rms.inpt_elem[i].itm_idx], 1);
		double tmp_pb = rand() / double(RAND_MAX)*1.0;
		int tmp_usr, tmp_itm;
		tmp_usr = tmp_itm = -1;
		if (tanh(tmp_resi*tmp_resi)>tmp_pb){
			if (i == last_pos[cur_rms.inpt_elem[i].usr_idx] || i == last_pos[cur_rms.inpt_elem[i].itm_idx + U_NUM]){
				tmp_usr = cur_rms.inpt_elem[i].usr_idx;
				tmp_itm = cur_rms.inpt_elem[i].itm_idx;
				sam_num+=mdf_cmb(tmp_usr, tmp_itm, cur_rms.inpt_elem[i].rat, cur_fms, cur_rms, cur_ss);
			}	
		    cur_rms.Buld_Rela(cur_rms.inpt_elem[i].usr_idx, cur_rms.inpt_elem[i].itm_idx + U_NUM, cur_rms.inpt_elem[i].rat);
		    cur_rms.Buld_Rela(cur_rms.inpt_elem[i].itm_idx + U_NUM, cur_rms.inpt_elem[i].usr_idx, cur_rms.inpt_elem[i].rat);
		}
		i++;
	}
    printf("%d\t%d\t", cur_rms.rela_num / 2, sam_num);
	/*for (int i = 0; i < FI_SI; i++)
		mdf_cmb(cur_rms.inpt_elem[i].usr_idx, cur_rms.inpt_elem[i].itm_idx, cur_rms.inpt_elem[i].rat, cur_fms, cur_rms, cur_ss);*/
	return 0;
}
