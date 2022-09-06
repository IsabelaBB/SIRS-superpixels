
# Similarity between Image and Reconstruction from Superpixels (SIRS)

Implementation of Similarity between Image and Reconstruction from Superpixels (SIRS). SIRS uses an RGB Bucket Descriptor (RBD) for each superpixel to measures the segmentation quality based on the exponential error of the reconstruction. For image reconstruction, SIRS selects the RBD color most similar to each pixel. Then, a variation of the mean-squared error, the Mean Exponential Error (MEE) expresses the reconstruction error between the original and reconstructed image. Finally, SIRS defines segmentation quality as the Gaussian weighted error of reconstruction using MEE.

### Requirements
The project was developed in **C/C++** under a **Linux-based** operational system; therefore, it is **NOT GUARANTEED** to work properly in other systems (_e.g._ Windows and macOS). It's also required to install OpenCV 4.
        
### Compiling and cleaning
- To compile all files: `make`
- For removing all generated files from source: `make clean`

### Running
Usage: `./bin/SIRS [OPTIONS]`

Options:
```
--img 		:	Original image file/path
--label 	: 	Segmented image file/path
--ext 		: 	Extension of segmented image (defaut: pgm)
--buckets 	: 	Number of color subsets (default:16)
--alpha 	: 	Number of subsets used to represent a superpixel (default:4)
--metric 	: 	Algorithm used to compute color homogeneity {1:SIRS, 2:EV} (default:1)
--gaussVar 	: 	Variance of gaussian in SIRS evaluation (default: 0.01)
--imgScores 	: 	File/Path of the colored result of color homogeneity (optional)
--drawScores 	: 	Boolean option {0,1} to write scores in the colored image result (imgScores option) (optional)
--log   	: 	txt log file with the mean value for a directory (optional)
--dlog 		: 	txt log file with the value of all images (optional)
--recon 	: 	File/Path of image reconstruction (optional)
```
**Examples:**
- Simple example: `./bin/SIRS --img ./image.jpg --label ./label_500.pgm --imgScores ./result.png`
- Example with image scores: `./bin/SIRS --img ./image.jpg --label ./label_100.pgm --imgScores ./result.png --drawScores 1`

## Cite
If this work was useful for your research, please cite our paper:
```
@InProceedings{barcelos2022improving,
  title={Improving color homogeneity measure in superpixel segmentation assessment},
  author={Barcelos, Isabela Borlido and Bel{\'e}m, Felipe and Melo, Leonardo and Falc{\~a}o, Alexandre Xavier and Guimar{\~a}es, Silvio Jamil F},
  booktitle={2022 35nd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},
  pages={},
  year={2022},
  organization={IEEE}
}
```

