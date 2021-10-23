#pragma ACCEL kernel

void kernel_doitgen(int nr,int nq,int np,double A[50][40][60],double C4[60][60],double sum[60])
{
  int r;
  int q;
  int p;
  int s;
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (r = 0; r < 50; r++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L1}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
    for (q = 0; q < 40; q++) {
      
#pragma ACCEL PIPELINE auto{__PIPE__L2}
      
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
      for (p = 0; p < 60; p++) {
        sum[p] = 0.0;
        
#pragma ACCEL PARALLEL reduction=sum FACTOR=auto{__PARA__L4}
        for (s = 0; s < 60; s++) {
          sum[p] += A[r][q][s] * C4[s][p];
        }
      }
      
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
      for (p = 0; p < 60; p++) {
        A[r][q][p] = sum[p];
      }
    }
  }
}
