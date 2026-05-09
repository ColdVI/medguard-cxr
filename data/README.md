# Data Directory

Place dataset documentation, checksums, and local download instructions here.

Do not commit raw patient data, images, DICOM files, or dataset archives.

## NIH ChestX-ray14

TBD.

## VinDr-CXR

TBD.

## RSNA Pneumonia Detection Challenge

This is the lower-friction Phase 3 localization fallback when VinDr-CXR is not
available locally. Do not commit the dataset files.

Expected local layout:

```text
data/rsna/
├── stage_2_train_labels.csv
├── stage_2_detailed_class_info.csv
└── stage_2_train_images/
    ├── <patientId>.dcm
    └── ...
```

The RSNA adapter uses `stage_2_train_labels.csv` bounding boxes
(`patientId,x,y,width,height,Target`) and normalizes them to `[0,1]` coordinates.
Only `Target=1` lung-opacity boxes are used for localization by default.

Access is through Kaggle/RSNA dataset terms, not PhysioNet credentialing. The
data is still medical imaging data, so keep it local, respect the provider
terms, and do not treat smoke or course-project results as clinical evidence.
