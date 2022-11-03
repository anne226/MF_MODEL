# MF_MODEL
before run **preprocess.py** you need to make sure your data follow the above format:  
>data  
  >>patient_name  
  >>>patient1_LOW_ENERGY_CC_R.dcm     
  patient1_RECOMBINED_CC_R.dcm  
  patient1_LOW_ENERGY_CC_L.dcm  
  patient1_RECOMBINED_CC_L.dcm  
  patient1_LOW_ENERGY_MLO_R.dcm  
  patient1_RECOMBINED_MLO_R.dcm  
  patient1_LOW_ENERGY_MLO_L.dcm  
  patient1_RECOMBINED_MLO_L.dcm  

**Note:** The DICOM files were renamed in order to match the corresponding names in excel  

 
     run **preprocess.py** to get the cropped, normalized images and saved as h5py.  
     
     run **train_fusion(concat).py** to train  
     
     run **test(concat).py** to test the performance  
      
*Required Libraries:  
 
     os, cv2, math, numpy, glob, h5py, skimage, pydicom, xlrd, torch, PIL, datetime, torchvision, matplotlib.
