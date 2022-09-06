#include "Image.h"

//=============================================================================
// Constructors & Deconstructors
//=============================================================================
Image *createImage(int num_rows, int num_cols, int num_channels)
{
    Image *new_img;

    new_img = (Image *)calloc(1, sizeof(Image));

    new_img->num_rows = num_rows;
    new_img->num_cols = num_cols;
    new_img->num_pixels = num_rows * num_cols;
    new_img->num_channels = num_channels;

    new_img->val = (int **)calloc(new_img->num_pixels, sizeof(int *));
#pragma omp parallel for
    for (int i = 0; i < new_img->num_pixels; i++)
        new_img->val[i] = (int *)calloc(num_channels, sizeof(int));

    return new_img;
}

void freeImage(Image **img)
{
    if (*img != NULL)
    {
        Image *tmp;

        tmp = *img;

        for (int i = 0; i < tmp->num_pixels; i++)
            free(tmp->val[i]);
        free(tmp->val);

        free(tmp);

        *img = NULL;
    }
}

NodeAdj *create4NeighAdj()
{
    NodeAdj *adj_rel;

    adj_rel = (NodeAdj *)calloc(1, sizeof(NodeAdj));

    adj_rel->size = 4;
    adj_rel->dx = (int *)calloc(4, sizeof(int));
    adj_rel->dy = (int *)calloc(4, sizeof(int));

    adj_rel->dx[0] = -1;
    adj_rel->dy[0] = 0; // Left
    adj_rel->dx[1] = 1;
    adj_rel->dy[1] = 0; // Right

    adj_rel->dx[2] = 0;
    adj_rel->dy[2] = -1; // Top
    adj_rel->dx[3] = 0;
    adj_rel->dy[3] = 1; // Bottom

    return adj_rel;
}

NodeAdj *create8NeighAdj()
{
    NodeAdj *adj_rel;

    adj_rel = (NodeAdj *)calloc(1, sizeof(NodeAdj));
    adj_rel->size = 8;
    adj_rel->dx = (int *)calloc(8, sizeof(int));
    adj_rel->dy = (int *)calloc(8, sizeof(int));

    adj_rel->dx[0] = -1;
    adj_rel->dy[0] = 0; // Center-Left
    adj_rel->dx[1] = 1;
    adj_rel->dy[1] = 0; // Center-Right

    adj_rel->dx[2] = 0;
    adj_rel->dy[2] = -1; // Top-Center
    adj_rel->dx[3] = 0;
    adj_rel->dy[3] = 1; // Bottom-Center

    adj_rel->dx[4] = -1;
    adj_rel->dy[4] = 1; // Bottom-Left
    adj_rel->dx[5] = 1;
    adj_rel->dy[5] = -1; // Top-Right

    adj_rel->dx[6] = -1;
    adj_rel->dy[6] = -1; // Top-Left
    adj_rel->dx[7] = 1;
    adj_rel->dy[7] = 1; // Bottom-Right
    return adj_rel;
}

void freeNodeAdj(NodeAdj **adj_rel)
{
    if (*adj_rel != NULL)
    {
        NodeAdj *tmp;

        tmp = *adj_rel;

        free(tmp->dx);
        free(tmp->dy);
        free(tmp);

        *adj_rel = NULL;
    }
}

//=============================================================================

int getMaximumValue(Image *img, int channel)
{
    int max_val, chn_begin, chn_end;

    max_val = -1;

    if (channel == -1)
    {
        chn_begin = 0;
        chn_end = img->num_channels - 1;
    }
    else
        chn_begin = chn_end = channel;

    for (int i = 0; i < img->num_pixels; i++)
        for (int j = chn_begin; j <= chn_end; j++)
            if (max_val < img->val[i][j])
                max_val = img->val[i][j];

    return max_val;
}

int getMinimumValue(Image *img, int channel)
{
    int min_val, chn_begin, chn_end;

    min_val = -1;

    if (channel == -1)
    {
        chn_begin = 0;
        chn_end = img->num_channels - 1;
    }
    else
        chn_begin = chn_end = channel;

    for (int i = 0; i < img->num_pixels; i++)
        for (int j = chn_begin; j <= chn_end; j++)
            if (min_val == -1 || min_val > img->val[i][j])
                min_val = img->val[i][j];

    return min_val;
}

int getNormValue(Image *img)
{
    int max_val;

    max_val = getMaximumValue(img, -1);

    if (max_val > 65535)
        printError("getNormValue", "This code supports only 8-bit and 16-bit images!");

    if (max_val <= 255)
        return 255;
    else
        return 65535;
}

inline bool areValidNodeCoordsImage(int num_rows, int num_cols, NodeCoords coords)
{
    return (coords.x >= 0 && coords.x < num_cols) &&
           (coords.y >= 0 && coords.y < num_rows);
}

inline int getNodeIndexImage(int num_cols, NodeCoords coords)
{
    return coords.y * num_cols + coords.x;
}

inline int getIndexImage(int num_cols, int row_index, int col_index)
{
    return row_index * num_cols + col_index;
}

inline double euclDistance(float *feat1, float *feat2, int num_feats)
{
    double dist;

    dist = 0;

    for (int i = 0; i < num_feats; i++)
        dist += (feat1[i] - feat2[i]) * (feat1[i] - feat2[i]);
    dist = sqrtf(dist);

    return dist;
}

inline double euclDistanceCoords(NodeCoords feat1, NodeCoords feat2)
{
    double dist;

    dist = 0;

    dist += ((float)feat1.x - (float)feat2.x) * ((float)feat1.x - (float)feat2.x);
    dist += ((float)feat1.y - (float)feat2.y) * ((float)feat1.y - (float)feat2.y);
    dist = sqrtf(dist);

    return dist;
}

inline NodeCoords getAdjacentNodeCoords(NodeAdj *adj_rel, NodeCoords coords, int id)
{
    NodeCoords adj_coords;

    adj_coords.x = coords.x + adj_rel->dx[id];
    adj_coords.y = coords.y + adj_rel->dy[id];

    return adj_coords;
}

inline NodeCoords getNodeCoordsImage(int num_cols, int index)
{
    NodeCoords coords;

    coords.x = index % num_cols;
    coords.y = index / num_cols;

    return coords;
}

//=============================================================================
// PGM images
//=============================================================================

void SkipComments(FILE *fp)
{
    int ch;
    char line[100];
    while ((ch = fgetc(fp)) != EOF && isspace(ch))
    {
        ;
    }

    if (ch == '#')
    {
        fgets(line, sizeof(line), fp);
        SkipComments(fp);
    }
    else
    {
        fseek(fp, -1, SEEK_CUR);
    }
}

// for reading:
int *readPGM_bkp(const char *file_name)
{
    FILE *pgmFile;
    int *labeledImage;
    char version[3];
    int i;
    int lo, hi;
    int num_rows, num_cols, max_gray;

    pgmFile = fopen(file_name, "rb");
    if (pgmFile == NULL)
    {
        perror("cannot open file to read");
        exit(EXIT_FAILURE);
    }
    fgets(version, sizeof(version), pgmFile);
    if (strcmp(version, "P2"))
    {
        fprintf(stderr, "Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &num_cols);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &num_rows);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &max_gray);
    fgetc(pgmFile);

    printf("Max gray:%d\n", max_gray);
    labeledImage = (int *)malloc(num_rows * num_cols * sizeof(int));

    if (max_gray > 255)
    {
        for (i = 0; i < num_rows * num_cols; i++)
        {
            hi = fgetc(pgmFile);
            lo = fgetc(pgmFile);
            labeledImage[i] = (hi << 8) + lo;
        }
    }
    else
    {
        for (i = 0; i < num_rows * num_cols; i++)
        {
            fscanf(pgmFile, "%d ", &lo);
            labeledImage[i] = lo;
            // printf("%d ", lo);
        }
    }

    fclose(pgmFile);
    return labeledImage;
}

int *readPGM(const char *file_name)
{
    FILE *pgmFile;
    int *labeledImage;
    char version[3];
    int i;
    int lo, hi;
    int num_rows, num_cols, max_gray;
    bool p5 = false;

    pgmFile = fopen(file_name, "rb");
    if (pgmFile == NULL)
    {
        perror("cannot open file to read");
        exit(EXIT_FAILURE);
    }
    fgets(version, sizeof(version), pgmFile);
    if (strcmp(version, "P2"))
    {
        p5 = true;
        if (strcmp(version, "P5"))
        {

            fprintf(stderr, "Wrong file type!\n");
            exit(EXIT_FAILURE);
        }
    }
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &num_cols);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &num_rows);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &max_gray);
    fgetc(pgmFile);

    printf("Max gray:%d\n", max_gray);
    labeledImage = (int *)malloc(num_rows * num_cols * sizeof(int));

    if (max_gray > 255)
    {
        for (i = 0; i < num_rows * num_cols; i++)
        {
            hi = fgetc(pgmFile);
            lo = fgetc(pgmFile);
            labeledImage[i] = (hi << 8) + lo;
        }
    }
    else
    {
        if (p5)
        {
            for (i = 0; i < num_rows; ++i)
            {
                for (int j = 0; j < num_cols; ++j)
                {
                    lo = fgetc(pgmFile);
                    labeledImage[i] = lo;
                }
            }
        }
        else
        {
            for (i = 0; i < num_rows * num_cols; i++)
            {
                fscanf(pgmFile, "%d ", &lo);
                labeledImage[i] = lo;
                // printf("%d ", lo);
            }
        }
    }

    fclose(pgmFile);
    return labeledImage;
}

/*
// and for writing
void writePGM(const char *filename, const PGMData *data)
{
    FILE *pgmFile;
    int i, j;
    int hi, lo;

    pgmFile = fopen(filename, "wb");
    if (pgmFile == NULL) {
        perror("cannot open file to write");
        exit(EXIT_FAILURE);
    }

    fprintf(pgmFile, "P5 ");
    fprintf(pgmFile, "%d %d ", data->col, data->row);
    fprintf(pgmFile, "%d ", data->max_gray);

    if (data->max_gray > 255) {
        for (i = 0; i < data->row; ++i) {
            for (j = 0; j < data->col; ++j) {
                hi = HI(data->matrix[i][j]);
                lo = LO(data->matrix[i][j]);
                fputc(hi, pgmFile);
                fputc(lo, pgmFile);
            }

        }
    }
    else {
        for (i = 0; i < data->row; ++i) {
            for (j = 0; j < data->col; ++j) {
                lo = LO(data->matrix[i][j]);
                fputc(lo, pgmFile);
            }
        }
    }

    fclose(pgmFile);
    deallocate_dynamic_matrix(data->matrix, data->row);
}

*/



