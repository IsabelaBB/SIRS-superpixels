
#include "PrioQueue.h"
#include "Image.h"
#include "Utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "ift.h"
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace cv;
using namespace std;

typedef struct TextsConfig
{
    Vec3b textColor;
    char text[10];
} TextsConfig;

typedef struct Args
{
    char *img_path, *label_path, *label_ext;
    char *imgScoresPath, *imgRecon;
    char *logFile, *dLogFile;
    int buckets, alpha, metric;
    bool drawScores;
    double gauss_variance;
} Args;

bool initArgs(Args *args, int argc, char *argv[])
{
    char *nbucketsChar, *alphaChar, *metricChar, *drawScoresChar, *gauss_varianceChar;

    args->img_path = parseArgs(argv, argc, "--img");
    args->label_path = parseArgs(argv, argc, "--label");
    args->label_ext = parseArgs(argv, argc, "--ext");
    args->logFile = parseArgs(argv, argc, "--log");
    args->dLogFile = parseArgs(argv, argc, "--dlog");
    args->imgScoresPath = parseArgs(argv, argc, "--imgScores");
    args->imgRecon = parseArgs(argv, argc, "--recon");

    nbucketsChar = parseArgs(argv, argc, "--buckets");
    alphaChar = parseArgs(argv, argc, "--alpha");
    metricChar = parseArgs(argv, argc, "--metric");
    drawScoresChar = parseArgs(argv, argc, "--drawScores");
    gauss_varianceChar = parseArgs(argv, argc, "--gaussVar");

    args->buckets = strcmp(nbucketsChar, "-") != 0 ? atoi(nbucketsChar) : 16;
    args->alpha = strcmp(alphaChar, "-") != 0 ? atoi(alphaChar) : 4;
    args->metric = strcmp(metricChar, "-") != 0 ? atoi(metricChar) : 1;
    args->drawScores = strcmp(drawScoresChar, "-") != 0 ? atoi(drawScoresChar) : false;
    args->gauss_variance = strcmp(gauss_varianceChar, "-") != 0 ? atof(gauss_varianceChar) : 0.01;
    
    if (strcmp(args->logFile, "-") == 0)
        args->logFile = NULL;
    if (strcmp(args->dLogFile, "-") == 0)
        args->dLogFile = NULL;
    if (strcmp(args->imgScoresPath, "-") == 0)
        args->imgScoresPath = NULL;
    if (strcmp(args->imgRecon, "-") == 0)
        args->imgRecon = NULL;
    if (strcmp(args->label_ext, "-") == 0)
        args->label_ext = "pgm";
    

    if (strcmp(args->img_path, "-") == 0 || strcmp(args->label_path, "-") == 0 || strcmp(args->label_ext, "-") == 0 || (args->metric == 1 && (args->buckets < 1 || args->alpha < 1 || args->alpha > args->buckets * 7)) || args->metric < 1)
        return false;

    return true;
}

Image *loadImage(const char *filepath)
{
    int num_channels, num_rows, num_cols;
    unsigned char *data;
    Image *new_img;

    data = stbi_load(filepath, &num_cols, &num_rows, &num_channels, 0);

    if (data == NULL)
        printError("loadImage", "Could not load the image <%s>", filepath);

    new_img = createImage(num_rows, num_cols, num_channels);

#pragma omp parallel for
    for (int i = 0; i < new_img->num_pixels; i++)
    {
        new_img->val[i] = (int *)calloc(new_img->num_channels, sizeof(int));

        for (int j = 0; j < new_img->num_channels; j++)
            new_img->val[i][j] = data[i * new_img->num_channels + j];
    }

    stbi_image_free(data);

    return new_img;
}

int *loadImageLabels(const char *filepath)
{
    int num_channels, num_rows, num_cols;
    unsigned char *data;
    int *new_img;

    data = stbi_load(filepath, &num_cols, &num_rows, &num_channels, 0);

    if (data == NULL)
        printError("loadImage", "Could not load the image <%s>", filepath);

    new_img = (int *)malloc(num_rows * num_cols * sizeof(int));

    for (int i = 0; i < num_cols * num_rows; i++)
    {
        int color = 0;
        for (int j = 0; j < num_channels; j++)
            color += data[i * num_channels + j];

        new_img[i] = color / num_channels;
    }

    stbi_image_free(data);

    return new_img;
}

void writeImagePGM(Image *img, const char *filepath)
{
    int max_val, min_val;
    FILE *fp;

    fp = fopen(filepath, "wb");

    if (fp == NULL)
        printError("writeImagePGM", "Could not open the file <%s>", filepath);

    max_val = getMaximumValue(img, -1);
    min_val = getMinimumValue(img, -1);

    fprintf(fp, "P5\n");
    fprintf(fp, "%d %d\n", img->num_cols, img->num_rows);
    fprintf(fp, "%d\n", max_val);

    // 8-bit PGM file
    if (max_val < 256 && min_val >= 0)
    {
        unsigned char *data;

        data = (unsigned char *)calloc(img->num_pixels, sizeof(unsigned char));

        for (int i = 0; i < img->num_pixels; i++)
            data[i] = (unsigned char)img->val[i][0];

        fwrite(data, sizeof(unsigned char), img->num_pixels, fp);

        free(data);
    }
    // 16-bit PGM file
    else if (max_val < 65536 && min_val >= 0)
    {
        unsigned short *data;

        data = (unsigned short *)calloc(img->num_pixels, sizeof(unsigned short));

        for (int i = 0; i < img->num_pixels; i++)
            data[i] = (unsigned short)img->val[i][0];

        for (int i = 0; i < img->num_pixels; i++)
        {
            int high, low;

            high = ((data[i]) & 0x0000FF00) >> 8;
            low = (data[i]) & 0x000000FF;

            fputc(high, fp);
            fputc(low, fp);
        }

        free(data);
    }
    else
        printError("writeImagePGM", "Invalid min/max spel values <%d,%d>", min_val, max_val);

    fclose(fp);
}

int filter(const struct dirent *name)
{
    int pos = 0;
    while (name->d_name[pos] != '.')
    {
        if (name->d_name[pos] == '\0')
            return 0;
        pos++;
    }
    pos++;
    if (name->d_name[pos] == '\0')
        return 0;

    int extSize = 0;
    while (name->d_name[pos + extSize] != '\0')
    {
        extSize++;
    }

    char ext[extSize];
    int pos2 = 0;
    while (pos2 < extSize)
    {
        ext[pos2] = name->d_name[pos + pos2];
        pos2++;
    }

    if ((extSize == 3 && ((ext[0] == 'p' && ext[1] == 'n' && ext[2] == 'g') ||
                          (ext[0] == 'j' && ext[1] == 'p' && ext[2] == 'g') ||
                          (ext[0] == 'p' && ext[1] == 'p' && ext[2] == 'm') ||
                          (ext[0] == 'p' && ext[1] == 'g' && ext[2] == 'm') ||
                          (ext[0] == 'b' && ext[1] == 'm' && ext[2] == 'p'))) ||
        (extSize == 4 && ext[0] == 't' && ext[1] == 'i' && ext[2] == 'f' && ext[2] == 'f'))
        return 1;

    return 0;
}

int getNumSuperpixels(int *L, int num_pixels)
{
    int tmp[num_pixels], numLabels = 0;

    for (int i = 0; i < num_pixels; i++)
        tmp[i] = 0;
    for (int i = 0; i < num_pixels; i++)
        tmp[L[i]] = -1;
    for (int i = 0; i < num_pixels; i++)
        if (tmp[i] == -1)
            numLabels++;

#ifdef DEBUG
    printf("Number of superpixels: %d\n", numLabels);
#endif

    return numLabels;
}

//==========================================================

void RBD(Image *image, int *labels, int label, int nbuckets, int *alpha, float **Descriptor/*, float *error_reference, int errType*/)
{
    /* Compute the superpixels descriptors
        image : RGB image
        labels     : Image labels (0,K-1)
        Descriptor : Descriptor[num_channels][alpha]
        Error       : (optional) Descriptor[num_channels][alpha]
        errType : {0:mode, 1:median}
    */

    PrioQueue *queue;
    int num_histograms = pow(2, image->num_channels) - 1;
    long int ColorHistogram[num_histograms][nbuckets][image->num_channels]; // Descriptor[image->num_channels][nbuckets]
    double V[num_histograms * nbuckets];                                    // buckets priority : V[image->num_channels][nbuckets]
    int superpixel_size;

    superpixel_size = 0;

    queue = createPrioQueue(num_histograms * nbuckets, V, MINVAL_POLICY);

    for (int h = 0; h < num_histograms; h++)
    {
        for (int b = 0; b < nbuckets; b++)
        {
            V[h * nbuckets + b] = 0;
            for (int c = 0; c < image->num_channels; c++)
                ColorHistogram[h][b][c] = 0;
        }
    }

    // compute histograms
    for (int i = 0; i < image->num_pixels; i++)
    {
        if (labels[i] == label)
        {
            superpixel_size++;

            //   R  G  B    GR BG  B R RGB
            // 110 101 011 100 001 010 000

            //   R  G  B    GR BG  B R RGB
            // 001 010 100 011 001 101 111

            //    R   G   B   D    |   GR  BG   B R |  BGR | D  R D G  DB   | D GR DBG  DB R  | DBGR
            // 1110 1101 1011 0111 | 1100 1001 1010 | 1000 | 0110 0101 0011 | 0100 0001 0010  | 0000
            int hist_id = 0;
            int bit = 1;

            // printf("(%d) hist_id:%d, bit:%d \n", i, hist_id, bit);
            for (int c1 = 0; c1 < image->num_channels; c1++)
            {
                for (int c2 = 0; c2 < image->num_channels; c2++)
                {
                    if (image->val[i][c1] < image->val[i][c2])
                    {
                        hist_id = hist_id | bit;
                        c2 = image->num_channels;
                    }
                }
                bit = bit << 1;
            }
            hist_id = ~hist_id; // indice da cor é a quantidade de shifts para encontrar cada bit 1
            hist_id *= -1;      // 100
            // printf("(i=%d) hist_id:%d, bit:%d \n", i, hist_id, bit);

            //   R  G  B     GR  BG   B R  BGR
            // 001 010 100  011  110  101  111  >> indice da cor é a quantidade de shifts

            // find one max channel index
            int max_channel = 0;
            int tmp_hist_id = hist_id;
            while (tmp_hist_id % 2 == 0)
            {
                tmp_hist_id = tmp_hist_id >> 1;
                max_channel++;
            }

            int bin = floor(((float)image->val[i][max_channel] / 255.0) * nbuckets);
            hist_id--;

            V[hist_id * nbuckets + bin]++;

            for (int c = 0; c < image->num_channels; c++)
                ColorHistogram[hist_id][bin][c] += (long int)image->val[i][c];
        }
    }

    for (int c = 0; c < num_histograms; c++)
    {
        for (int b = 0; b < nbuckets; b++)
        {
            if (V[c * nbuckets + b] > 0)
            {
                if (isPrioQueueEmpty(queue) || queue->last_elem_pos < (*alpha) - 1)
                    insertPrioQueue(&queue, c * nbuckets + b); // push (color, frequency) into Q, sorted by V[i]
                else
                {
                    if (!isPrioQueueEmpty(queue) && V[queue->node[0]] < V[c * nbuckets + b])
                    {
                        popPrioQueue(&queue);
                        insertPrioQueue(&queue, c * nbuckets + b); // push (color, frequency) into Q, sorted by V[i]
                    }
                }
            }
        }
    }

    // Get the higher alpha buckets
    if (isPrioQueueEmpty(queue))
    {
        (*alpha) = 0;
        int a = 0;

        for (int c = 0; c < image->num_channels; c++)
            Descriptor[a][c] = 0;
    }
    else
    {
        if (queue->last_elem_pos < (*alpha) - 1)
            (*alpha) = queue->last_elem_pos + 1;

        for (int a = (*alpha) - 1; a >= 0; a--)
        {
            int val = popPrioQueue(&queue);
            int bin = val % nbuckets;
            int hist = (val - bin) / nbuckets;

            for (int c = 0; c < image->num_channels; c++)
            {
                Descriptor[a][c] = (float)ColorHistogram[hist][bin][c] / (float)V[val]; // get the mean color
            }
        }
    }

    freePrioQueue(&queue);
}

double *SIRS(int *labels, Image *image, int alpha, int nbuckets, char *reconFile, double gauss_variance, double *score)
{

    double *histogramVariation;
    float ***Descriptor;     
    int *descriptor_size;    
    int emptySuperpixels;
    double **MSE;
    Mat recons;
    double **variation_descriptor; 

    (*score) = 0;
    emptySuperpixels = 0;

    recons = Mat::zeros(image->num_rows, image->num_cols, CV_8UC3);

    int superpixels = 0;
    for (int i = 0; i < image->num_pixels; ++i)
    {
        if (labels[i] > superpixels)
            superpixels = labels[i];
    }
    superpixels++;

    histogramVariation = (double *)calloc(superpixels, sizeof(double));
    descriptor_size = (int *)calloc(superpixels, sizeof(int));
    Descriptor = (float ***)calloc(superpixels, sizeof(float **));
    MSE = (double **)calloc(superpixels, sizeof(double *));
    
    variation_descriptor = (double **)calloc(superpixels, sizeof(double *));

    std::vector<int> superpixelSize(superpixels, 0);
    for (int i = 0; i < image->num_pixels; ++i)
        superpixelSize[labels[i]]++;

    for (int s = 0; s < superpixels; s++)
    {
        double *mean_buckets;     

        if (superpixelSize[s] == 0)
            emptySuperpixels++;

        descriptor_size[s] = alpha;
        Descriptor[s] = (float **)calloc(alpha, sizeof(float *));
        MSE[s] = (double *)calloc(image->num_channels, sizeof(double));
        mean_buckets = (double *)calloc(image->num_channels, sizeof(double));
        variation_descriptor[s] = (double *)calloc(image->num_channels, sizeof(double));

        for (int c = 0; c < alpha; c++)
            Descriptor[s][c] = (float *)calloc(image->num_channels, sizeof(float));

        RBD(image, labels, s, nbuckets, &(descriptor_size[s]), Descriptor[s]);

        for (int i = 0; i < image->num_channels; i++)
            MSE[s][i] = 0.0;

        for (int a = 0; a < descriptor_size[s]; a++)
        {
            for (int c = 0; c < image->num_channels; c++)
                mean_buckets[c] += ((double)Descriptor[s][a][c] / 255.0);
        }
        for (int c = 0; c < image->num_channels; c++)
            mean_buckets[c] /= descriptor_size[s];

        for (int a = 0; a < descriptor_size[s]; a++)
        {
            for (int c = 0; c < image->num_channels; c++)
                variation_descriptor[s][c] = MAX(variation_descriptor[s][c], abs(((double)Descriptor[s][a][c] / 255.0) - mean_buckets[c]));
        }

        free(mean_buckets);
    }

    for (int i = 0; i < image->num_pixels; i++)
    {
        int label;
        NodeCoords coords;
        double minVariance;
        int descIndex;

        label = labels[i];
        coords = getNodeCoordsImage(image->num_cols, i);
        Vec3b &recons_color = recons.at<Vec3b>(coords.y, coords.x);

        descIndex = 0;
        minVariance = 0;

        for (int c = 0; c < image->num_channels; c++)
            minVariance += ((double)image->val[i][c] / 255.0 - (double)Descriptor[label][0][c] / 255.0) * ((double)image->val[i][c] / 255.0 - (double)Descriptor[label][0][c] / 255.0);

        // find the most distance descriptor values
        for (int h1 = 1; h1 < descriptor_size[label]; h1++)
        {
            double val = 0;
            
            for (int c = 0; c < image->num_channels; c++)
                val += ((double)image->val[i][c] / 255.0 - (double)Descriptor[label][h1][c] / 255.0) * ((double)image->val[i][c] / 255.0 - (double)Descriptor[label][h1][c] / 255.0);

            if (val < minVariance)
            {
                minVariance = val;
                descIndex = h1;
            }
        }

        if (reconFile != NULL)
        {
            for (int c = 0; c < image->num_channels; c++)
                recons_color[image->num_channels - 1 - c] = Descriptor[label][descIndex][c];
        }
        for (int c = 0; c < image->num_channels; c++)
        {
            MSE[label][c] += pow(abs((double)image->val[i][c] / 255.0 - (double)Descriptor[label][descIndex][c] / 255.0), 2 - variation_descriptor[label][c]);
        }
    }

    double sum_MSE[image->num_channels];

    for (int c = 0; c < image->num_channels; c++)
        sum_MSE[c] = 0.0;

#ifdef DEBUG
        printf("\n\nScores:\n");
#endif

    for (int s = 0; s < superpixels; s++)
    {
        histogramVariation[s] = 0;
        for (int c = 0; c < image->num_channels; c++)
        {
            histogramVariation[s] += exp(-(MSE[s][c] / superpixelSize[s]) / gauss_variance); 
            sum_MSE[c] += MSE[s][c];                                                         
        }
        histogramVariation[s] /= image->num_channels;

#ifdef DEBUG
        printf("%f, ", histogramVariation[s]);
#endif
    }

    (*score) = 0;
    for (int c = 0; c < image->num_channels; c++)
    {
        (*score) += sum_MSE[c];
    }
    (*score) /= image->num_channels;
    (*score) = exp(-((*score) / image->num_pixels) / (gauss_variance));

#ifdef DEBUG
    printf("\nImage score = %f \n", (*score));
#endif

    for (int s = 0; s < superpixels; s++)
    {
        for (int a = 0; a < descriptor_size[s]; a++)
        {
            free(Descriptor[s][a]);
        }
        free(Descriptor[s]);
        free(MSE[s]);
        free(variation_descriptor[s]);
    }
    free(Descriptor);
    free(descriptor_size);
    free(MSE);
    free(variation_descriptor);

    if (reconFile != NULL)
        imwrite(reconFile, recons);

    return histogramVariation;
}

double *explainedVariation(int *labels, Image *image, char *reconFile, double *score)
{
    (*score) = 0;

    double *supExplainedVariation;
    double *valuesTop, *valuesBottom;
    Mat recons;

    // get the higher label
    int superpixels = 0;
    for (int i = 0; i < image->num_pixels; ++i)
    {
        if (labels[i] > superpixels)
            superpixels = labels[i];
    }
    superpixels++;

    recons = Mat::zeros(image->num_rows, image->num_cols, CV_8UC3);

    supExplainedVariation = (double *)calloc(superpixels, sizeof(double));
    valuesTop = (double *)calloc(superpixels, sizeof(double));
    valuesBottom = (double *)calloc(superpixels, sizeof(double));

    std::vector<cv::Vec3f> mean(superpixels, cv::Vec3f(0, 0, 0));
    std::vector<cv::Vec3f> squared_mean(superpixels, cv::Vec3f(0, 0, 0));
    std::vector<int> count(superpixels, 0);

    cv::Vec3f overall_mean = 0;
    for (int i = 0; i < image->num_pixels; ++i)
    {
        for (int c = 0; c < image->num_channels; ++c)
        {
            mean[labels[i]][c] += image->val[i][c];
            overall_mean[c] += image->val[i][c];
        }
        count[labels[i]]++;
    }

    for (int i = 0; i < superpixels; ++i)
    {
        valuesTop[i] = 0;
        valuesBottom[i] = 0;
        for (int c = 0; c < image->num_channels; ++c)
            mean[i][c] /= count[i];
    }

    overall_mean /= image->num_rows * image->num_cols;

    for (int i = 0; i < image->num_pixels; ++i)
    {
        int label;
        NodeCoords coords;

        label = labels[i];
        coords = getNodeCoordsImage(image->num_cols, i);
        Vec3b &recons_color = recons.at<Vec3b>(coords.y, coords.x);

        for (int c = 0; c < image->num_channels; ++c)
        {
            valuesTop[label] += (mean[label][c] - overall_mean[c]) * (mean[labels[i]][c] - overall_mean[c]);
            valuesBottom[label] += (image->val[i][c] - overall_mean[c]) * (image->val[i][c] - overall_mean[c]);
            if (reconFile != NULL)
                recons_color[image->num_channels - 1 - c] = mean[labels[i]][c];
        }
    }

    double sum_top = 0;
    double sum_bottom = 0;

#ifdef DEBUG
    printf("\n\nScores:\n");
#endif
    for (int s = 0; s < superpixels; s++)
    {
        sum_top += valuesTop[s];
        sum_bottom += valuesBottom[s];

        if (valuesBottom[s] == 0)
            supExplainedVariation[s] = 1;
        else
            supExplainedVariation[s] = valuesTop[s] / valuesBottom[s];

#ifdef DEBUG
        printf("%f + ", supExplainedVariation[s]);
#endif
    }

    (*score) += sum_top / sum_bottom;

#ifdef DEBUG
    printf("= %f \n", (*score));
#endif

    free(valuesTop);
    free(valuesBottom);

    if (reconFile != NULL)
        imwrite(reconFile, recons);

    // return sum_top/sum_bottom;
    return supExplainedVariation;
}

//==========================================================

void drawtorect(cv::Mat &mat, cv::Rect target, int face, int thickness, cv::Scalar color, const std::string &str)
{
    Size rect = getTextSize(str, face, 1.0, thickness, 0);

    target.width -= 8;
    target.height -= 8;
    target.x += 4;
    target.y += 4;

    // if(target.width < 43 || target.height < 15) return;
    if (target.height < 10 || ((float)target.width / (float)target.height < 1.8 && target.width < 36))
        return;
    if (target.width > 60)
    {
        target.x += (target.width - 60) / 2;
        target.width = 60;
    }
    if (target.height > 13)
    {
        target.y += (target.height - 13) / 2;
        target.height = 13;
    }

    double scalex = (double)target.width / (double)rect.width;
    double scaley = (double)target.height / (double)rect.height;
    double scale = min(scalex, scaley);
    int marginx = scale == scalex ? 0 : (int)((double)target.width * (scalex - scale) / scalex * 0.5);
    int marginy = scale == scaley ? 0 : (int)((double)target.height * (scaley - scale) / scaley * 0.5);

    putText(mat, str, Point(target.x + marginx, target.y + target.height - marginy), face, scale, color, thickness, 8, false);
}

Rect findMinRect(const Mat1b &src)
{
    Mat1f W(src.rows, src.cols, float(0));
    Mat1f H(src.rows, src.cols, float(0));

    Rect maxRect(0, 0, 0, 0);
    float maxArea = 0.f;

    for (int r = 0; r < src.rows; ++r)
    {
        for (int c = 0; c < src.cols; ++c)
        {
            if (src(r, c) == 0)
            {
                H(r, c) = 1.f + ((r > 0) ? H(r - 1, c) : 0);
                W(r, c) = 1.f + ((c > 0) ? W(r, c - 1) : 0);
            }

            float minw = W(r, c);
            for (int h = 0; h < H(r, c); ++h)
            {
                minw = min(minw, W(r - h, c));
                float area = (h + 1) * minw;
                if (area > maxArea)
                {
                    maxArea = area;
                    maxRect = Rect(Point(c - minw + 1, r - h), Point(c + 1, r + 1));
                }
            }
        }
    }

    return maxRect;
}

void createImageMetric(int *L, double *colorVariance, int num_rows, int num_cols, const char *filename, bool showScores)
{

    NodeAdj *AdjRel;
    Mat image;
    int K;

    K = 0;
    for (int p = 0; p < num_rows * num_cols; p++)
        K = MAX(K, L[p]);
    K++;

    int superpixel_size[K];
    TextsConfig textsConfig[K];
    vector<vector<Point>> contours(K);
    int count_contours[K];

    AdjRel = create8NeighAdj();
    image = Mat::zeros(num_rows, num_cols, CV_8UC3);

    int max_Label = 0;
    for (int s = 0; s < K; s++)
    {
        superpixel_size[s] = 0;
        count_contours[s] = 0;
    }
    for (int p = 0; p < num_rows * num_cols; p++)
    {
        superpixel_size[L[p]]++;
        max_Label = MAX(max_Label, L[p]);
    }

    for (int s = 0; s < K; s++)
        contours[s] = vector<Point>(superpixel_size[s]);

    for (int p = 0; p < num_rows * num_cols; p++)
    {
        Vec3b color, border;
        NodeCoords coords;
        int label = L[p];

        color[0] = color[1] = color[2] = (int)(255 * MIN(1, colorVariance[label])); // get superpixel color according to its error
        border[0] = border[1] = border[2] = (color[0] < 128) ? 255 : 0;

        bool isBorder = false;
        coords = getNodeCoordsImage(num_cols, p);

        for (int j = 0; j < AdjRel->size; j++)
        {
            NodeCoords adj_coords;
            adj_coords = getAdjacentNodeCoords(AdjRel, coords, j);

            if (areValidNodeCoordsImage(num_rows, num_cols, adj_coords))
            {
                int adj_index;
                adj_index = getNodeIndexImage(num_cols, adj_coords);
                if (label != L[adj_index])
                {
                    isBorder = true;
                    if (image.at<Vec3b>(Point(adj_coords.x, adj_coords.y))[0] == 0 && image.at<Vec3b>(Point(adj_coords.x, adj_coords.y))[0] == 255)
                        image.at<Vec3b>(Point(coords.x, coords.y)) = border;
                    else
                        image.at<Vec3b>(Point(coords.x, coords.y)) = image.at<Vec3b>(Point(adj_coords.x, adj_coords.y));
                    break;
                }
            }
        }
        if (!isBorder)
            image.at<Vec3b>(Point(coords.x, coords.y)) = color;

        contours[label][count_contours[label]] = Point(coords.x, coords.y);
        count_contours[label]++;

        gcvt(colorVariance[label], 2, textsConfig[label].text);
        textsConfig[label].textColor[0] = color[0] < 128 ? 255 : 0;
        textsConfig[label].textColor[1] = textsConfig[label].textColor[2] = textsConfig[label].textColor[0];
    }

    // Apply the colormap
    applyColorMap(image, image, COLORMAP_OCEAN);

    if (showScores)
    {
        vector<vector<Point>> contours_poly(contours.size());
        vector<Rect> boundRect(contours.size());

        for (size_t i = 0; i < contours.size(); ++i)
        {
            if (contours[i].size() > 0)
            {
                // Create a mask for each single blob
                Mat1b maskSingleContour(num_rows, num_cols, uchar(0));
                
                for (size_t j = 0; j < contours[i].size(); j++)
                {
                    maskSingleContour.at<uchar>(contours[i][j]) = 255;
                }

                // Find minimum rect for each blob
                Rect box = findMinRect(~maskSingleContour);

                // Draw rect
                drawtorect(image, box, cv::FONT_HERSHEY_PLAIN, 1, textsConfig[i].textColor, textsConfig[i].text);
            }
        }
    }
    imwrite(filename, image);
    freeNodeAdj(&AdjRel);
}


//==========================================================

// computeMetric(image_name, label_file, imgScores, args, &numSuperpixels);
double computeMetric(char *img_path, char *labels_path, char *imgScores, char *reconFile, Args args, int *numSuperpixels)
{
    iftImage *labels;
    Image *image;
    int *labeledImage;
    double *homogeneityScores, score;
    int num_rows, num_cols;

    score = 0;

    labels = iftReadImageByExt(labels_path);

    num_cols = labels->xsize;
    num_rows = labels->ysize;

    labeledImage = (int *)malloc(labels->n * sizeof(int));
    int maxLabel = 0, minLabel = INT_MAX;
    for (int i = 0; i < labels->n; i++)
    {
        labeledImage[i] = labels->val[i];
        if (maxLabel < labeledImage[i])
            maxLabel = labeledImage[i];
        if (minLabel > labeledImage[i])
            minLabel = labeledImage[i];
    }

    // color homogeneity measures: original img and labels
    if (args.metric < 4)
    {
        image = loadImage(img_path);
    }

    iftDestroyImage(&labels);

    (*numSuperpixels) = getNumSuperpixels(labeledImage, num_cols * num_rows);

#ifdef DEBUG
    printf("Measure segm. error: ");
#endif

    switch (args.metric)
    {
    // ========================================
    // COLOR-BASED MEASURES
    // ========================================
    case 1:
#ifdef DEBUG
        printf("SIRS \n");
#endif
        homogeneityScores = SIRS(labeledImage, image, args.alpha, args.buckets, reconFile, args.gauss_variance, &score);
    break;

    case 2:
#ifdef DEBUG
        printf("EV \n");
#endif
        homogeneityScores = explainedVariation(labeledImage, image, reconFile, &score);
        break;

    default:
        break;
    }

    if (imgScores != NULL)
        createImageMetric(labeledImage, homogeneityScores, image->num_rows, image->num_cols, imgScores, args.drawScores);

    freeImage(&image);
    if (args.metric < 3 && homogeneityScores != NULL)
        free(homogeneityScores);
    
    free(labeledImage);
    return score;
}

void runDirectory(Args args)
{
    // determine mode : file or path
    struct stat sb;
    char label_file[255], imgScores[255], reconFile[255];
    double score_path = 0;
    int numSuperpixels;

#ifdef DEBUG
    printf("Init runDirectory\n");
#endif

    if (stat(args.img_path, &sb) == -1)
    {
        perror("stat");
        exit(EXIT_SUCCESS);
    }

    int type;
    switch (sb.st_mode & S_IFMT)
    {
    case S_IFDIR:
        printf("directory processing\n");
        type = 0;
        break;
    case S_IFREG:
        printf("single file processing\n");
        type = 1;
        break;
    default:
        type = -1;
        break;
    }

    if (type == -1)
        exit(EXIT_SUCCESS);
    else if (type == 1)
    {

        // ALLOC
        double score_img = 0;

        // RUN CODE: 'img_path' is a file name
        score_img = computeMetric(args.img_path, args.label_path, args.imgScoresPath, args.imgRecon, args, &numSuperpixels);

        // ************************

#ifdef DEBUG
    printf("Score: %f , %d superpixels \n", score_img, numSuperpixels);
#endif
        
    }
    else if (type == 0)
    {
        // get file list
        struct dirent **namelist;
        int n = scandir(args.img_path, &namelist, &filter, alphasort);
        if (n == -1)
        {
            perror("scandir");
            exit(EXIT_FAILURE);
        }

#ifdef DEBUG
    printf(" %i image(s) found\n", n);
#endif
        if (n == 0)
            exit(EXIT_SUCCESS);

        // process file list
        char *image_name = (char *)malloc(255);

        int numImages = n;
        int sum_num_superpixel = 0;

        while (n--)
        {
            // get image name
            sprintf(image_name, "%s/%s", args.img_path, namelist[n]->d_name);

            // RUN CODE: 'img_path' is a directory

            // alloc structures
            int end = 0;
            while (namelist[n]->d_name[end] != '\0')
            {
                end++;
            }
            while (namelist[n]->d_name[end] != '.')
            {
                end--;
            }

            char fileName[255];
            double score_img = 0;
            strncpy(fileName, namelist[n]->d_name, end);
            fileName[end] = '\0';

            if (args.imgScoresPath != NULL)
                sprintf(imgScores, "%s/%s.png", args.imgScoresPath, fileName);

            if (args.imgRecon != NULL)
                sprintf(reconFile, "%s/%s_recon.png", args.imgRecon, fileName);

            sprintf(label_file, "%s/%s.%s", args.label_path, fileName, args.label_ext);

            // ********

            // run code

            // RUN CODE: 'img_path' is a file name
            if (args.imgScoresPath == NULL)
            {
                if (args.imgRecon == NULL)
                    score_img = computeMetric(image_name, label_file, NULL, NULL, args, &numSuperpixels);
                else
                    score_img = computeMetric(image_name, label_file, NULL, reconFile, args, &numSuperpixels);
            }
            else
            {
                if (args.imgRecon == NULL)
                    score_img = computeMetric(image_name, label_file, imgScores, NULL, args, &numSuperpixels);
                else
                    score_img = computeMetric(image_name, label_file, imgScores, reconFile, args, &numSuperpixels);
            }

            // ********
            sum_num_superpixel += numSuperpixels;
            score_path += score_img;
            if (args.dLogFile != NULL)
            {
                FILE *fp = fopen(args.dLogFile, "a+");
                fprintf(fp, "%s %d %.5f\n", namelist[n]->d_name, numSuperpixels, score_img);
                fclose(fp);
            }

            // ********

            free(namelist[n]);
        }

#ifdef DEBUG
    printf("Superpixels: %f, Mean score: %f \n", (double)sum_num_superpixel / (double)numImages, score_path / (double)numImages);
#endif
        if (args.logFile != NULL)
        {
            FILE *fp = fopen(args.logFile, "a+");
            fprintf(fp, "%.5f %.5f\n", (double)sum_num_superpixel / (double)numImages, score_path / (double)numImages);

            fclose(fp);
            free(image_name);
            free(namelist);
        }

        // Free the rest of structures

        // ************************
    }
}

int main(int argc, char *argv[])
{
    /*
    args:
        img         :   RGB image
        label       :   A pgm file with labeled image
        ext         :   Extension of labels image path (defaut: pgm)
        buckets     :   Number of buckets
        alpha       :   Number of buckets used to represent a superpixel
        metric      :   Algorithm used to compute color homogeneity {1:SIRS, 2:EV}
        gaussVar    :   Variance of SIRS evaluation (default: 0.01)
        imgScores   :   Path to write the labeled image according to the color homogeneity (optional)
        drawScores  :   Boolean option {0,1} to write scores in the output image (imgScores option) (optional)
        log         :   txt log file with the mean value for a directory (optional)
        dlog        :   txt log file with the value of all images (optional)
        recon       :   Directory to image reconstruction (optional)
    */

#ifdef DEBUG
    printf("DEGUB true\n");
#endif

    Args args;

    int initReturn = initArgs(&args, argc, argv);

    if (!initReturn)
    {
        printf("--img img_path --label img_labels --ext extLabels --buckets nbuckets --alpha alpha --metric metric [--log logFile --dlog detailedLogFile --imgScores imgScores_dir --drawScores drawScores] \n\n");
        return 0;
    }

    runDirectory(args);

    return 0;
}
