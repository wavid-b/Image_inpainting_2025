# Image Processing Project - Image Inpainting 
### Spring 2025

## Outline of Report 
1. Introduction to Image Inpainting problem
2. How images were obfuscated 
3. Binary classifier (SVM) 
4. Corruption Localization (You Only Look Once CNN) 
5. Inpainting algorithm comparison (Classical and ZITS transformer) 

## Organization of code by section 
### Obfuscation 
- img_splines.ipynb
- create_obfuscated.ipynb
### Binary Classifier
- binary_classifier.ipynb
- add_noise.ipynb
### Corruption Localization
- split_files.ipynb
- gen_bounding_box.ipynb
- Note - to run, a modified yolo5 was used and that is too large for github (I ran out of LFS)
### Inpainting 
- Some approaches in section_3.ipynb
- Lama in lama_inpainting.py

  
