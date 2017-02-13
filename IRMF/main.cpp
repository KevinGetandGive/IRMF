#include<ctime>
#include<iostream>
#include<omp.h>
#include<cstdio>
#include<cstring>
#include"Global_Config.h"
#include"Global_Function.h"
#include"Storage_Structure.h"
using namespace std;
ftr_mtx_strc fms;
rat_mtx_strc rms;
stttc_strc ss;
bool sam_used[R_NUM];
int main(){
	int head_tmst, tail_tmst;
	int rat_num;
	head_tmst = clock();
	Glb_Init(fms, rms, ss);
	rat_num = Glb_Inpt_Init(0, fms, rms, ss);
	rms.Push_Data(rat_num);
	Batch_Train(fms, rms, ss);
	for (int i = 1; i < DEPARTS_NUM; i++){
		long long t1 = clock();
		rat_num = Glb_Inpt_Init(i, fms, rms, ss);
		Incre_Train(rat_num,fms, rms, ss);
		long long t2 = clock();
        long long dett=(t2-t1)*1000/CLOCKS_PER_SEC;
		printf("%lld\n",  dett);

	}
	return 0;
}
