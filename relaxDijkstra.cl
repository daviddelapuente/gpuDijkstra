__kernel void relaxDijkstra(__global int* G,__global int* dinfo)
{	
	int id=get_global_id(0);
	int n=dinfo[0];
	//F[id]==1
	if(dinfo[id*4+1+2]==1){
		//preguntamos por todos sus vecinos
		for(int j=0;j<n;j++){
			if(G[j+id*n]>0){
				//son vecinos
				//U[j]==1
				if(dinfo[j*4+2]==1){
					//delta[j]
					dinfo[j*4+2+2]=min(dinfo[j*4+2+2], dinfo[id*4+2+2] + G[j + id * n]);
				}
			}
		}
	}
}