# ***Generalized Deep Convolutional Neuroal Network (gDCNN) for artifact rejection in MEG***


## **Motivation**

In contrast to [1], this routine is capable of processing 4D-Neuroimaging, CTF and Neuromag data. It performs ICA on chunks of MEG data (2-4 minutes) and apply an automatic labeling (cardiac or ocular) for all ICA components in all chops of data.
The information is used for training a generalised version of the DCNN model (gDCNN).

[1] Ahmad Hasasneh, Nikolas Kampel, Praveen Sripad, N. Jon Shah, and Juergen Dammers. *Deep Learning Approach for Automatic Classification of Ocular and Cardiac Artifacts in MEG Data.* Journal of Engineering, vol. 2018, Article ID 1350692,10 pages, 2018.
https://doi.org/10.1155/2018/1350692
    
## **Remarks - about data:**
* Data should not be filtered (recommendation).
* **It is assumed that bad channels are marked as bad.** It is highly recommended to replace bad channels with an interpolated signal.
* **MEG system specific noise reduction should be applied (e.g., sss) prior to this analysis**
* **ECG and EOG channels must be of good quality.** Horizontal EOG can be used in addition to vertical EOG.
* Raw continuous data should be used (no epochs)
* Besides resting state also task data should be used for training
* Equal numbers of female/male would be nice

## **Remarks - about config:**
* MEG system specs must be defined in config file
* ECG and EOG channel names must be provided and fit to all data to be processed.
* In cases of changes in AUX names for different raw files you must use a different config file
* for different data sets you may want to use different directories


## **Workflow**
1. Compute ICA on filtered and down sampled data. Note, cleaning will be applied on unfiltered data later. (or raw data that has been filtered differently)
2. Identification of ECG and EOG components using standard methods as implemented in MNE-Python (CTPS and correlation)
3. Results will be save in as dict in numpy format. Only ICA components together relevant information about the data (DCNN object) is stored
4. Data is used to train the gDCNN model (in Juelich)
5. gDCNN-based artifact rejection ist tested against standard methods for all MEG systems


