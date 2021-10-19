#include "merlin_type_define.h"
#pragma ACCEL kernel

void kernel_2mm(int ni,int nj,int nk,int nl,double alpha,double beta,double tmp[180][190],double A[180][210],double B[210][190],double C[190][220],double D[180][220])
{
  int i;
  int j;
  int k;
/* D := alpha*A*B*C + beta*D */
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 180; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L2}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
    for (j = 0; j < 190; j++) {
      tmp[i][j] = 0.0;
      
#pragma ACCEL PARALLEL reduction=tmp FACTOR=auto{__PARA__L4}
      for (k = 0; k < 210; ++k) {
        tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (i = 0; i < 180; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    for (j = 0; j < 220; j++) {
      D[i][j] *= beta;
      
#pragma ACCEL PARALLEL reduction=D FACTOR=auto{__PARA__L5}
      for (k = 0; k < 190; ++k) {
        D[i][j] += tmp[i][k] * C[k][j];
      }
    }
  }
}
