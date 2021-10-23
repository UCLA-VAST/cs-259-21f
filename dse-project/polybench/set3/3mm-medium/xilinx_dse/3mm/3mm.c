
#pragma ACCEL kernel

void kernel_3mm(int ni,int nj,int nk,int nl,int nm,double E[180][190],double A[180][200],double B[200][190],double F[190][210],double C[190][220],double D[220][210],double G[180][210])
{
  int i;
  int j;
  int k;
/* E := A*B */
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (i = 0; i < 180; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L3}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L3}
    for (j = 0; j < 190; j++) {
      E[i][j] = 0.0;
      
#pragma ACCEL PARALLEL reduction=E FACTOR=auto{__PARA__L6}
      for (k = 0; k < 200; ++k) {
        E[i][j] += A[i][k] * B[k][j];
      }
    }
  }
/* F := C*D */
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L1}
  for (i = 0; i < 190; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L4}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L4}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L4}
    for (j = 0; j < 210; j++) {
      F[i][j] = 0.0;
      
#pragma ACCEL PARALLEL reduction=F FACTOR=auto{__PARA__L7}
      for (k = 0; k < 220; ++k) {
        F[i][j] += C[i][k] * D[k][j];
      }
    }
  }
/* G := E*F */
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L2}
  for (i = 0; i < 180; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L5}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L5}
    
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L5}
    for (j = 0; j < 210; j++) {
      G[i][j] = 0.0;
      
#pragma ACCEL PARALLEL reduction=G FACTOR=auto{__PARA__L8}
      for (k = 0; k < 190; ++k) {
        G[i][j] += E[i][k] * F[k][j];
      }
    }
  }
}
