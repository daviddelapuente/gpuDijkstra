__kernel void updateDijkstra(__global int* dinfo)
{	
	int id=get_global_id(0);
	//F[id]=0
	dinfo[id*4+1+2]=0;
	//U[id]==1 && delta[id]<=T
	int T=dinfo[1];
	if(dinfo[id*4+2]==1 && dinfo[id*4+2+2]<=T){
		//U[id]=0
		dinfo[id*4+2]=0;
		//F[id]=1
		dinfo[id*4+1+2]=1;
	}
}