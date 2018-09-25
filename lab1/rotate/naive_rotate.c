#include <stdlib.h>
#include <time.h>
#include <stdio.h>

//#define DEBUG

typedef short pixel;

#define RIDX(i, j, n) ((i)*(n)+(j))

void naive_rotate(int dim, pixel *src, pixel *dst)
{
	int i, j;
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			dst[RIDX(dim-1-j, i, dim)] = src[RIDX(i, j, dim)];
}

// 1,567,682 N^2 / 32 + N^2 / 4
void naive_rotate_1(int dim, pixel *src, pixel *dst)
{
	int i, j, ii, jj;
	for (ii = 0; ii < dim; ii+=4)
		for (jj = 0; jj < dim; jj+=4)
			for (i = ii; i < ii+4; i++)
				for (j = jj; j < jj+4; j++)
					dst[RIDX(dim-1-j, i, dim)] = src[RIDX(i, j, dim)];
}

// 1,426,463 N^2 / 32 + N^2 / 32
void naive_rotate_2(int dim, pixel *src, pixel *dst)
{
	int i, j, ii, jj;
	for (ii = 0; ii < dim; ii+=32)
		for (jj = 0; jj < dim; jj+=32)
			for (i = ii; i < ii+32; i+=4)
				for (j = jj; j < jj+32; j+=4) {
					dst[RIDX(dim-1-j,i,dim)] = src[RIDX(i,j,dim)];
					dst[RIDX(dim-1-j,i+1,dim)] = src[RIDX(i+1,j,dim)];
					dst[RIDX(dim-1-j,i+2,dim)] = src[RIDX(i+2,j,dim)];
					dst[RIDX(dim-1-j,i+3,dim)] = src[RIDX(i+3,j,dim)];

					dst[RIDX(dim-1-j-1,i,dim)] = src[RIDX(i,j+1,dim)];
					dst[RIDX(dim-1-j-1,i+1,dim)] = src[RIDX(i+1,j+1,dim)];
					dst[RIDX(dim-1-j-1,i+2,dim)] = src[RIDX(i+2,j+1,dim)];
					dst[RIDX(dim-1-j-2,i+3,dim)] = src[RIDX(i+3,j+1,dim)];

					dst[RIDX(dim-1-j-2,i,dim)] = src[RIDX(i,j+2,dim)];
					dst[RIDX(dim-1-j-2,i+1,dim)] = src[RIDX(i+1,j+2,dim)];
					dst[RIDX(dim-1-j-2,i+2,dim)] = src[RIDX(i+2,j+2,dim)];
					dst[RIDX(dim-1-j-2,i+3,dim)] = src[RIDX(i+3,j+2,dim)];

					dst[RIDX(dim-1-j-3,i,dim)] = src[RIDX(i,j+3,dim)];
					dst[RIDX(dim-1-j-3,i+1,dim)] = src[RIDX(i+1,j+3,dim)];
					dst[RIDX(dim-1-j-3,i+2,dim)] = src[RIDX(i+2,j+3,dim)];
					dst[RIDX(dim-1-j-3,i+3,dim)] = src[RIDX(i+3,j+3,dim)];
				}
}

// 1,562,835 N^2 / 32 + N^2 / 4
void naive_rotate_3(int dim, pixel *src, pixel *dst)
{
	int i, j, ii, jj;
	for (ii = 0; ii < dim; ii+=4)
		for (jj = 0; jj < dim; jj+=4)
			for (i = ii; i < ii+4; i++) {
				for (j = jj + 1; j < jj+4; j++)
					dst[RIDX(dim-1-j, i, dim)] = src[RIDX(i, j, dim)];
				dst[RIDX(dim-1-jj, i, dim)] = src[RIDX(i, jj, dim)];
			}
}

// 635,912 N^2 / 32 + N^2 / 32
#define COPY(d,s) *(d) = *(s)
void naive_rotate_4(int dim, pixel *src, pixel *dst)
{
	int i, j;
	for (i = 0; i < dim; i+=32)
		for (j = dim-1; j >= 0; j--) {
			pixel *dptr = dst+RIDX(dim-1-j,i,dim);
			pixel *sptr = src+RIDX(i,j,dim);
			COPY(dptr, sptr); sptr += dim;      COPY(dptr+1, sptr); sptr += dim;
			COPY(dptr+2, sptr); sptr += dim;    COPY(dptr+3, sptr); sptr += dim;
			COPY(dptr+4, sptr); sptr += dim;    COPY(dptr+5, sptr); sptr += dim;
			COPY(dptr+6, sptr); sptr += dim;    COPY(dptr+7, sptr); sptr += dim;
			COPY(dptr+8, sptr); sptr += dim;    COPY(dptr+9, sptr); sptr += dim;
			COPY(dptr+10, sptr); sptr += dim;   COPY(dptr+11, sptr); sptr += dim;
			COPY(dptr+12, sptr); sptr += dim;   COPY(dptr+13, sptr); sptr += dim;
			COPY(dptr+14, sptr); sptr += dim;   COPY(dptr+15, sptr); sptr += dim;
			COPY(dptr+16, sptr); sptr += dim;   COPY(dptr+17, sptr); sptr += dim;
			COPY(dptr+18, sptr); sptr += dim;   COPY(dptr+19, sptr); sptr += dim;
			COPY(dptr+20, sptr); sptr += dim;   COPY(dptr+21, sptr); sptr += dim;
			COPY(dptr+22, sptr); sptr += dim;   COPY(dptr+23, sptr); sptr += dim;
			COPY(dptr+24, sptr); sptr += dim;   COPY(dptr+25, sptr); sptr += dim;
			COPY(dptr+26, sptr); sptr += dim;   COPY(dptr+27, sptr); sptr += dim;
			COPY(dptr+28, sptr); sptr += dim;   COPY(dptr+29, sptr); sptr += dim;
			COPY(dptr+30, sptr); sptr += dim;   COPY(dptr+31, sptr);
		}

}

#define N 2048
pixel A[N*N], B[N*N];

int main(int argc, const char* argv[]) {
	if(argc == 2) {
		switch(argv[1][0]) {
			case '1': {
				naive_rotate_1(N, A, B);
				break;
			}
			case '2': {
				naive_rotate_2(N, A, B);
				break;
			}
			case '3': {
				naive_rotate_3(N, A, B);
				break;
			}
			case '4': {
				naive_rotate_4(N, A, B);
				break;
			}
			default: {
				naive_rotate(N, A, B);
				break;
			}
		}
	}
	return 0;
}
