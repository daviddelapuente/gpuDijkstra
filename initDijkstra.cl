__kernel void initDijkstra(__global int* G,__global int* dinfo)
{	
	int id=get_global_id(0);
	if(id==0){
		//U
		dinfo[id*4+2]=0;
		//F
		dinfo[id*4+1+2]=1;
		//delta
		dinfo[id*4+2+2]=0;
	}else{
		//U
		dinfo[id*4+2]=1;
		//F
		dinfo[id*4+1+2]=0;
		//delta
		dinfo[id*4+2+2]=INT_MAX;
	}

	int n=dinfo[0];
	int mAux=INT_MAX;
	for(int j=0;j<n;j++){
		if (G[j + id * n] > 0) {
                    mAux = min(mAux, G[j + id * n]);
                }	
	}
	dinfo[id*4+3+2]=mAux;
}