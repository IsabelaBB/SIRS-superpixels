#ifndef IMAGE_H
#define IMAGE_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// HEADERS
//=============================================================================

#include "Utils.h"
#include "Color.h"

#include <float.h>
#include <limits.h>
#include <regex.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <assert.h>
#include <ctype.h>
#include <dirent.h>
#include <libgen.h>
#include <math.h>
#include <mm_malloc.h>
#include <stdarg.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

//=============================================================================
// Macros
//=============================================================================
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#define IFT_INFINITY_INT       INT_MAX
#define IFT_INFINITY_INT_NEG   INT_MIN
#define IFT_INFINITY_LONG      LONG_MAX
#define IFT_INFINITY_LONG_NEG  LONG_MIN
#define IFT_INFINITY_ULONG     ULONG_MAX
#define IFT_INFINITY_FLT       FLT_MAX
#define IFT_INFINITY_FLT_NEG  -FLT_MAX
#define IFT_INFINITY_DBL       DBL_MAX
#define IFT_INFINITY_DBL_NEG  -DBL_MAX
#define IFT_INFINITY_LDBL      LDBL_MAX
#define IFT_INFINITY_LDBL_NEG -LDBL_MAX

#define IFT_STR_DEFAULT_SIZE 4096
#define IFT_ULONG_NIL IFT_INFINITY_ULONG

#define IFT_INFINITY_FLT_NEG  -FLT_MAX

#define IFT_PI      3.14159265358979323846
#define IFT_WHITE       0
#define IFT_GRAY        1
#define IFT_BLACK       2
#define IFT_NIL        -1
#define IFT_INCREASING  1
#define IFT_DECREASING  0
#define IFT_EPSILON     1E-07

#define iftMax(x,y) (((x) > (y))?(x):(y))
#define iftMin(x,y) (((x) < (y))?(x):(y))
#define iftRound(x) ((x < 0)?(int)(x-0.5):(int)(x+0.5))
#define iftVoxelDistance(u,v) (sqrtf((u.x-v.x)*(u.x-v.x)+(u.y-v.y)*(u.y-v.y)+(u.z-v.z)*(u.z-v.z)))

#define WHITEPOINT_X	0.950456
#define WHITEPOINT_Y	1.0
#define WHITEPOINT_Z	1.088754

#define LABF(t)	\
	((t >= 8.85645167903563082e-3) ? \
	pow(t,0.333333333333333) : (841.0/108.0)*(t) + (4.0/29.0))

#if defined(__WIN32) || defined(__WIN64)
#define IFT_SEP_C "\\"
#else
#define IFT_SEP_C "/"
#endif

#define iftGetVoxelIndex(s, v) ((v.x)+(s)->tby[(v.y)]+(s)->tbz[(v.z)])
#define iftImgVal(img, x, y, z) img->val[((x)+(img)->tby[(y)]+(img)->tbz[(z)])]
#define iftImgCb(img, x, y, z) img->Cb[((x)+(img)->tby[(y)]+(img)->tbz[(z)])]
#define iftImgCr(img, x, y, z) img->Cr[((x)+(img)->tby[(y)]+(img)->tbz[(z)])]
#define iftImgAlpha(img, x, y, z) img->alpha[((x)+(img)->tby[(y)]+(img)->tbz[(z)])]
#define iftCopyVoxelSize(src, dst) (dst)->dx = (src)->dx; (dst)->dy = (src)->dy; (dst)->dz = (src)->dz;
#define iftValidVoxel(img, v)((v.x >= 0) && (v.x <= ((img)->xsize - 1)) && (v.y >= 0) && (v.y <= ((img)->ysize - 1)) && (v.z >= 0) && (v.z <= ((img)->zsize - 1)))
#define iftGetXCoord(s, p) (((p) % (((s)->xsize)*((s)->ysize))) % (s)->xsize)
#define iftGetYCoord(s, p) (((p) % (((s)->xsize)*((s)->ysize))) / (s)->xsize)
#define iftGetZCoord(s, p) ((p) / (((s)->xsize)*((s)->ysize)))
#define iftImgVoxelVal(i, v) i->val[iftGetVoxelIndex(i, v)]
#define iftImgVoxelCb(i, v) i->Cb[iftGetVoxelIndex(i, v)]
#define iftImgVoxelCr(i, v) i->Cr[iftGetVoxelIndex(i, v)]
#define iftDiagonalSize(s) (iftRound(sqrtf(s->xsize*s->xsize + s->ysize*s->ysize + s->zsize*s->zsize)))

#define iftGetMatrixCol(m,i) ((i) % (m)->ncols)
#define iftGetMatrixRow(m,i) ((i) / (m)->ncols)
#define iftGetMatrixIndex(m,c,r) ((c)+(m)->tbrow[(r)])
#define iftMatrixRowPointer(m, r) ((m)->val + iftGetMatrixIndex((m), 0, (r)))
#define iftMatrixElem(m, c, r) (m)->val[iftGetMatrixIndex((m), (c), (r))]

#define iftMGetVoxelIndex(s,v) ((v.x)+(s)->tby[(v.y)]+(s)->tbz[(v.z)])

#define MSG_MEMORY_ALLOC_ERROR  "Cannot allocate memory space"
#define MSG_FILE_OPEN_ERROR  "Cannot open file %s"

//=============================================================================
// TYPES
//=============================================================================

typedef struct ift_int_array iftIntArray;
typedef struct ift_matrix iftMatrix;

typedef struct timeval timer;
typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned long  ulong;
#ifndef  __cplusplus
typedef long long llong;
typedef unsigned long long ullong;
#endif

//=============================================================================
// Structures
//=============================================================================

typedef struct ift_band {
    float *val;
} iftBand;

typedef struct ift_vector {
    float x, y, z;
} iftVector, iftPoint;

typedef struct ift_voxel {
    int x, y, z;
} iftVoxel;

typedef struct ift_size {
    int x, y, z;
} iftSize;

typedef struct ift_interval {
    float begin;
    float end;
} iftInterval;

typedef struct ift_image_domain {
    int xsize;
    int ysize;
    int zsize;
} iftImageDomain;

typedef struct ift_voxel_size {
    float dx;
    float dy;
    float dz;
} iftVoxelSize;

struct ift_int_array {
    long n;
    int *val;
};

typedef struct ift_flt_array {
    long n;
    float *val;
} iftFloatArray;

typedef struct ift_adjrel {
    int *dx, *dy, *dz;
    int n;
} iftAdjRel;

typedef enum ift_color_space {
    YCbCr_CSPACE,
    YCbCrNorm_CSPACE,
    RGB_CSPACE,
    RGBNorm_CSPACE,
    GRAY_CSPACE,
    GRAYNorm_CSPACE,
    WEIGHTED_YCbCr_CSPACE,
    LAB_CSPACE,
    LABNorm_CSPACE,
    LABNorm2_CSPACE,
    HSV_CSPACE
} iftColorSpace;

typedef struct ift_color {
  int val[3];
  float alpha;
} iftColor;

typedef struct ift_fcolor {
	float val[3];
} iftFColor;

typedef struct ift_file {
    char *path;
    int label;
    int sample;
    char *suffix;
    char status;
} iftFile;

typedef struct ift_bounding_box {
    iftVoxel begin;
    iftVoxel end;
} iftBoundingBox;

typedef struct ift_bitmap {
    char *val;
    int nbytes;
    int n;
} iftBMap;

typedef struct ift_image {
    int *val;
    ushort *Cb;
    ushort *Cr;

    int *R;
    int *G;
    int *B;
    
    ushort *alpha;
    int xsize;
    int ysize;
    int zsize;
    float dx;
    float dy;
    float dz;
    int *tby, *tbz;
    int n;
} iftImage;

typedef struct ift_matrix {
    int nrows;
    int ncols;
    long n;
    float *val;
    long *tbrow;
    bool allocated;
} iftMatrix;

typedef struct ift_mimage {
    iftMatrix *data;
    float **val;
    int xsize,ysize,zsize;
    float dx,dy,dz; 
    int *tby, *tbz;
    int n,m;
} iftMImage;

typedef struct ift_kernel {
  iftAdjRel *A;
  float     *weight;
} iftKernel;

typedef struct ift_set {
    int elem;
    struct ift_set *next;
} iftSet;

typedef struct ift_snode {
    char *elem; 
    struct ift_snode *prev;
    struct ift_snode *next;
} iftSNode;

typedef struct ift_slist {
    long n;
    iftSNode *head;
    iftSNode *tail;
} iftSList;

typedef struct ift_dir {
    char *path;
    long nfiles;
    iftFile **files;
    long nsubdirs;
    struct ift_dir **subdirs;
} iftDir;

typedef struct ift_fileset {
    /** Number of Files */
    long n;
    /** Array of Files */
    iftFile **files;
} iftFileSet;

typedef struct ift_csv {
    /** Number of rows of the CSV matrix. */
    long nrows;
    /** Number of columns of the CSV matrix. */
    long ncols;
    /** Header of the CSV. */
    char **header;
    /** CSV matrix of strings. Each string has 512 characters. */
    char ***data;
} iftCSV;

//=============================================================================

void iftCopyVoxel(iftVoxel *src, iftVoxel *dst);
iftIntArray *iftCreateIntArray(long n);
void iftDestroyIntArray(iftIntArray **iarr);
void iftShuffleIntArray(int* array, int n);

iftFloatArray *iftCreateFloatArray(long n);
void iftDestroyFloatArray(iftFloatArray **darr);

float iftRandomUniform(float low, float high); 
int iftRandomInteger (int low, int high);
int *iftRandomIntegers(int low, int high, int nelems); 
void iftSkipComments(FILE *fp);
double iftLog(double val, double base);
long iftNormalizationValue(long maxval);
void iftRandomSeed(unsigned int);
timer *iftTic(void);
timer *iftToc(void);
float iftCompTime(timer *tic, timer *toc);

iftAdjRel *iftCreateAdjRel(int n);
void iftDestroyAdjRel(iftAdjRel **A);
iftAdjRel *iftSpheric(float r);
iftAdjRel *iftCircular(float r);
iftAdjRel *iftCopyAdjacency(const iftAdjRel *A);
iftVoxel iftGetAdjacentVoxel(const iftAdjRel *A, iftVoxel u, int adj);

iftColor iftRGBColor(int R, int G, int B);
iftColor  iftRGBtoYCbCrBT2020(iftColor cin, const int rgbBitDepth, const int yCbCrBitDepth);
iftColor iftRGBtoYCbCr(iftColor cin, int normalization_value);
iftColor iftYCbCrtoRGB(iftColor cin, int normalization_value);
iftColor  iftYCbCrBT2020toRGB(iftColor cin, const int yCbCrBitDepth, const int rgbBitDepth);
iftFColor iftRGBtoLabNorm(iftColor rgb, int normalization_value);
iftColor  iftRGBtoHSV(iftColor cin, int normalization_value);
iftColor  iftHSVtoRGB(iftColor cin, int normalization_value);

bool iftFileExists(const char *pathname);
const char *iftFileExt(const char *pathname);
char *iftJoinPathnames(long n, ...);
char *iftFilename(const char *pathname, const char *suffix);
void iftDestroyFile(iftFile **file);
bool iftIsImageFile(const char *img_pathname);
bool iftIsImagePathnameValid(const char *img_pathname);
iftFile *iftCreateFile(const char *format, ...);
char *iftExpandUser(const char *path);

iftBMap *iftCreateBMap(int n);
void iftDestroyBMap(iftBMap **bmap);

iftImage *iftReadImageByExt(const char *filename, ...);
iftImage *iftCreateImage(int xsize, int ysize, int zsize);
iftImage *iftCopyImage(const iftImage *img);
void iftDestroyImage(iftImage **img);
void iftWriteImageByExt(const iftImage *img, const char *filename, ...);

int iftMaximumValue(const iftImage *img);
int iftMinimumValue(const iftImage *img);

iftVoxel iftGetVoxelCoord(const iftImage *img, int p);
iftImage *iftSelectImageDomain(int xsize, int ysize, int zsize);
iftBoundingBox iftMinBoundingBox(const iftImage *img, iftVoxel *gc_out);
iftImage *iftReadImage(const char *filename, ...);
iftImage* iftReadImagePNG(const char* format, ...);
iftImage* iftReadImageJPEG(const char* format, ...);
iftImage *iftReadImageP5(const char *filename, ...);
iftImage *iftReadImageP6(const char *filename, ...);
iftImage *iftReadImageP2(const char *filename, ...);
iftImage  *iftCreateImageFromBuffer(int xsize,int ysize,int zsize, int *val);
void iftCopyImageInplace(const iftImage *src, iftImage *dst);
void iftWriteImage(const iftImage *img, const char *filename, ...);
void iftWriteImageP5(const iftImage *img, const char *filename, ...);
void iftWriteImageP6(const iftImage *img, const char *filename, ...);
void iftWriteImageP2(const iftImage *img, const char *filename, ...);
void iftWriteImagePNG(const iftImage* img, const char* format, ...);
void iftWriteImageJPEG(const iftImage* img, const char* format, ...);
int iftMaximumValueInRegion(const iftImage *img, iftBoundingBox bb);
void iftSetImage(iftImage *img, int value);
void  iftSetAlpha(iftImage *img, ushort value);
void iftSetCbCr(iftImage *img, ushort value);
void iftVerifyImageDomains(const iftImage *img1, const iftImage *img2, const char *function);
uchar iftImageDepth(const iftImage* img);

void iftMinMaxValues(const iftImage *img, int *min, int *max);
iftImage *iftCreateImageFromImage(const iftImage *src);
iftImage *iftCreateColorImage(int xsize, int ysize, int zsize, int depth);
void iftPutXYSlice(iftImage *img, const iftImage *slice, int zcoord);
iftImage *iftGetXYSlice(const iftImage *img, int zcoord);
iftImage *iftBMapToBinImage(const iftBMap *bmap, int xsize, int ysize, int zsize);
iftBMap *iftBinImageToBMap(const iftImage *bin_img);

void iftConvertNewBitDepth(iftImage **img, int new_depth);

iftMatrix *iftCreateMatrix(int ncols, int nrows);
iftMatrix *iftCopyMatrix(const iftMatrix *A);
void iftDestroyMatrix(iftMatrix **M);

iftMImage  *iftCreateMImage(int xsize,int ysize,int zsize, int nbands);
void iftDestroyMImage(iftMImage **img);
iftMImage *iftImageToMImage(const iftImage *img, char color_space);
iftImage *iftMImageToImage(const iftMImage *img, int Imax, int band);

iftVoxel    iftMGetVoxelCoord(const iftMImage *img, int p);
char iftMValidVoxel(const iftMImage *img, iftVoxel v);
float iftMMaximumValue(const iftMImage *img, int band);

iftKernel   *iftCreateKernel(iftAdjRel *A);
void         iftDestroyKernel(iftKernel **K);

int *iftAllocIntArray(long n);
void iftCopyIntArray(int *array_dst, const int *array_src, int nelems);
#ifndef  __cplusplus
long long *iftAllocLongLongIntArray(long n);
void iftCopyLongLongIntArray(long long *array_dst, const long long *array_src, int nelems);
#endif
float *iftAllocFloatArray(long n);
void iftCopyFloatArray(float *array_dst, float *array_src, int nelems);
double *iftAllocDoubleArray(long n);
void iftCopyDoubleArray(double *array_dst, double *array_src, int nelems);
ushort *iftAllocUShortArray(long n);
uchar *iftAllocUCharArray(long n);
char *iftAllocCharArray(long n);
char *iftAllocString(long n);
#define iftSwap(x, y) do { __typeof__(x) _IFT_SWAP_ = x; x = y; y = _IFT_SWAP_; } while (0)

void iftInsertSet(iftSet **S, int elem);
int iftRemoveSet(iftSet **S);
void    iftRemoveSetElem(iftSet **S, int elem);
void    iftDestroySet(iftSet **S);
iftSet* iftSetUnion(iftSet *S1,iftSet *S2);
iftSet* iftSetConcat(iftSet *S1,iftSet *S2);
char    iftUnionSetElem(iftSet **S, int elem);
void    iftInvertSet(iftSet **S);
int 	iftSetSize(const iftSet* S);
iftSet* iftSetCopy(iftSet* S);
int     iftSetHasElement(iftSet *S, int elem);
iftIntArray *iftSetToArray(iftSet *S);

iftSList *iftCreateSList();
void iftDestroySList(iftSList **SL);

void iftError(const char *msg, const char *func, ...);
void iftWarning(const char *msg, const char *func, ...);

char *iftGetLine(FILE *stream) ;

void iftRightTrim(char* s, char c);
iftSList *iftSplitString(const char *phrase, const char *delimiter);
char *iftLowerString(const char *str);
bool iftCompareStrings(const char *str1, const char *str2);
char *iftSplitStringAt(const char *phrase, const char *delimiter, long position);
char *iftCopyString(const char *str, ...);
char *iftRemoveSuffix(const char *str, const char *suffix);
bool iftEndsWith(const char *str, const char *suffix);
bool iftStartsWith(const char *str, const char *prefix);
char *iftConcatStrings(int n, ...);
char *iftRemovePrefix(const char *str, const char *prefix);
char *iftReplaceString(const char *str, const char *old_sub, const char *new_sub);

bool iftDirExists(const char *pathname, ...);
char *iftParentDir(const char *pathname);
void iftMakeDir(const char *dir_path);
iftDir *iftLoadFilesFromDirByRegex(const char *dir_pathname, const char *regex);
iftDir *iftLoadDir(const char *dir_pathname, long hier_levels);
void iftDestroyDir(iftDir **dir);

bool iftRegexMatch(const char *str, const char *regex_pattern, ...);
iftIntArray *iftIntRange(int begin, int end, int inc);
void iftFQuickSort( float *value, int *index, int i0, int i1, uchar order); 

iftFileSet *iftLoadFileSetFromDirOrCSV(const char *file_entry, long hier_levels, bool sort_pathnames);
iftFileSet *iftLoadFileSetFromCSV(const char *csv_pathname, bool sort_pathnames);
iftFileSet *iftLoadFileSetFromDir(const char *dir_pathname, long hier_level);
iftFileSet *iftCreateFileSet(long nfiles);
iftFileSet *iftLoadFileSetFromDirByRegex(const char *dir_pathname, const char *regex, bool sort_pathnames);
void iftDestroyFileSet(iftFileSet **farr);
void iftSortFileSet(iftFileSet *files); 
char *iftExpandUser(const char *path);
iftFile *iftCopyFile(const iftFile* file);

iftCSV *iftReadCSV(const char *csv_pathname, const char separator);
void iftDestroyCSV(iftCSV **csv);

void iftConvertNewBitDepth
(iftImage **img, int new_depth);
iftBMap *iftGetBorderMap
(const iftImage *label);


//=============================================================================
// FUNCTIONS
//=============================================================================
static inline iftFColor iftRGBtoLab(iftColor rgb, int normalization_value)
{
    /* with no normalization. */
    // L in [0, 99.998337]
    // a in [-86.182236, 98.258614]
    // b in [-107.867744, 94.481682]

    //RGB to XYZ
    float R = rgb.val[0]/(float)normalization_value;
    float G = rgb.val[1]/(float)normalization_value;
    float B = rgb.val[2]/(float)normalization_value;

    float X = (0.4123955889674142161*R + 0.3575834307637148171*G + 0.1804926473817015735*B);
    float Y = (0.2125862307855955516*R + 0.7151703037034108499*G + 0.07220049864333622685*B);
    float Z = (0.01929721549174694484*R + 0.1191838645808485318*G + 0.9504971251315797660*B);

    //XYZ to lab
    X /= WHITEPOINT_X;
    Y /= WHITEPOINT_Y;
    Z /= WHITEPOINT_Z;
    X = LABF(X);
    Y = LABF(Y);
    Z = LABF(Z);
    float L = 116*Y - 16;
    float a = 500*(X - Y);
    float b = 200*(Y - Z);

    iftFColor lab;
    lab.val[0] = L;
    lab.val[1] = a;
    lab.val[2] = b;

    return lab;
}

static inline iftFColor iftRGBtoLabNorm2(iftColor rgb, int normalization_value)
{
    /* get lab values*/
    iftFColor lab=iftRGBtoLab(rgb,normalization_value);
    /*normalize each value*/
    lab.val[0]=lab.val[0]/99.998337f;
    lab.val[1]=(lab.val[1]+86.182236f)/(86.182236f+98.258614f);
    lab.val[2]=(lab.val[2]+107.867744f)/(107.867744f+94.481682f);

    return lab;
}

static inline bool iftPathnameExists(const char *pathname) {
    struct stat buffer;
    return (stat (pathname, &buffer) == 0);
}

inline void iftBMapSet1(iftBMap *bmap, int b) {
    bmap->val[b >> 3] |= (1 << (b & 0x07));
}
inline bool iftBMapValue(const iftBMap *bmap, int b) {
    return ((bmap->val[b >> 3] & (1 << (b & 0x07))) != 0);
}

static inline bool iftIsColorImage(const iftImage *img) {
    return ((img->Cb != NULL) && (img->Cr != NULL));
}

static inline bool iftIs3DImage(const iftImage *img) {
    return (img->zsize > 1);
}

static inline long iftMaxImageRange(uchar img_depth) {
    return (1L << (img_depth)) - 1; // 2^img_depth -1
}

static inline bool iftIs3DMImage(const iftMImage *img) {
    return (img->zsize > 1);
}

static inline bool iftIsSListEmpty(const iftSList *SL) {
    return (SL->n == 0);
}

static inline void *iftAlloc(size_t n, size_t size) {
    return calloc(n, size);
}
static inline void iftFree(void *data) {
    if (data != NULL)
        free(data);
}


#ifdef __cplusplus
}
#endif

#endif // IMAGE_H