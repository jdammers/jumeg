jumeg
=====

MEG Data Analysis at FZJ

Warning
=======

1. Please do not push any changes directly to the master branch of jdammers/jumeg. Always make a pull request, discuss and only then merge with the master branch.
2. If you have code that you want to share and are not sure how to, please put it on a gist (https://gist.github.com/) and raise an issue on Github so that it can be discused and added.

User Installation (quick)
=========================

It is recommended to setup the mne-friendly jumeg environment using Anaconda / Miniconda.
After downloading anaconda, an environment can be easily created using the below steps:

1. Download conda environment file to download jumeg [default](https://gist.github.com/pravsripad/0361ffee14913487eb7b7ef43c9367fe) or [jumeg with GPU support](https://gist.github.com/pravsripad/7bb8f696999985b442d9aca8ade19f19).

2. Create the conda environment using the below commands.

``` conda env create -f=jumeg.yml ```

or 

``` conda env create -f=jumeg_cuda.yml ```

Developer Installation
======================

1. Fork the repository by visiting https://github.com/jdammers/jumeg and clicking on the Fork button. 
2. Your own fork of the repo will be created at https://github.com/your_username/jumeg.
3. Clone the repository from the system shell using the command:
   git clone https://github.com/your_username/jumeg
4. Add it to your local python site-packages using the below commands from the same directory as above:

   ```   site_dir=`python -c'import site; print(site.getusersitepackages())'` ```

   ```   present_dir=`pwd` ```
   
   ```   ln -s "${present_dir}/jumeg" "${site_dir}/jumeg"   ```
   
5. Now open the python/ipython shell and try "import jumeg" to check if it is correctly installed. 
   

Using GIT
=========

Add the main repository to your git using the below command:
   ```
   git remote add upstream https://github.com/jdammers/jumeg
   ```
Updating to the latest master version (where upstream is main repository - jdammers/jumeg):
   ```
   git checkout master
   git pull upstream master
   ```
Making some changes and updating it to the web server: 

1. Create a new branch for your changes:
   git checkout -b <new_branch>

2. Make your changes to the files, add new files etc. 

3. Check your changes using "git status". 

4. Add these changed files to git and commit it to the web server (where origin is your fork of the main repo eg. pravsripad/jumeg):
   ```
   git add <file1> <file2>
   git commit -m"Some useful comments" <file1> <file2>
   git push origin <new_branch>
   ```
5. Now go to ```https://github.com/<yourname>/jumeg``` and use the "Compare and pull request" option to raise a pull request where you can discuss and finally merge the new changes to main master branch. 

6. Use ```git checkout master``` and ```git pull upstream master``` to remain on the latest changes before making your own changes. 

Standard module and file naming conventions
===========================================

**Modules**

1. All modules should have a prefix of 'jumeg_' added. For example, jumeg_math.py, jumeg_io.py and so on. 

2. Subdirectories can be created with a relevant name and the files should follow the naming pattern. For example, the subdirectory ICA will contain jumeg_infomax.py, jumeg_fastica.py etc.

3. Please follow the coding conventions as used in mne-python as much as possible.

**Files**

Please follow the below file name conventions when writing and reading your code using the jumeg codebase. An example file name along with descriptions are shown below. 

1. Unchanged Raw File Name - ```109925_CAU01A_10-07-15@08:42_2_c,rfDC.fif```.
   
   (109925 - subject id, CAU01A - experiment id, 10-07-15 - date of measuement, 08:42 - time of measurement, c,rfDC - mode of measurement)

2. Exported raw data (from BTI to FIFF) - ```109925_CAU01A_100715_0842_2_c,rfDC-raw.fif```.
   All files with -raw.fif will contain both MEG and EEG/ECG/EOG channels combined. 

3. Exported raw data with only MEG channels - ```109925_CAU01A_100715_0842_2_c,rfDC-meg.fif```.

4. Exported raw data with only EEG/ECG/EOG channels - ```109925_CAU01A_100715_0842_2_c,rfDC-eeg.fif```.

5. Raw data band pass filtered from 1 to 45 Hz - ```109925_CAU01A_100715_0842_2_c,rfDC,fibp1-45-raw.fif```.
   
   (fibp - band pass filtered, filp - low pass filtered, fihp - high pass filtered, fin - notch filtered)

6. ICA object after decomposition - ```109925_CAU01A_100715_0842_2_c,rfDC,bp1-45-ica.fif```.
   
   (-ica suffix)

7. Raw file after artifact rejection (preprocessed and cleaned raw data) - ```109925_CAU01A_100715_0842_2_c,rfDC,bp1-45,ar-raw.fif```.
   
   (,ar added)

8. Averages of evoked files - ```109925_CAU01A_100715_0842_2_c,rfDC,bp1-45,ar,STI014_003-evoked.fif``` or
                              ```109925_CAU01A_100715_0842_2_c,rfDC,bp1-45,ar,happy-evoked.fif``` or
                              ```109925_CAU01A_100715_0842_2_c,rfDC,bp1-45,ar,STIMIX_MIX-evoked.fif```.
   
   The condition upon which the file is averaged should be mentioned. eg. happy, sad, audicued etc. If this is not available then the channel name and event id used to generate the averages should be added. If there are many conditions, then STIMIX_MIX should be used. 

9. Empty room files - ```109925_CAU01A_100715_0842_2_c,rfDC,bp1-45-empty.fif```.
   
   (suffix of -empty indicates that it is an empty room raw file)

10. If plots or figures are to be generated, they should always be in a separate subdirectory './plots' and the plot file names should have the exact same name as the raw file used to generate it, except for the file extension which should be different. 
      
   e.g. ```109925_CAU01A_100715_0842_2_c,rfDC,bp1-45,ar,happy-evoked.png``` would indicate a plot of averages over 'happy condition' for 1 to 45 Hz band passed data. 

11. Raw file after artifact rejection, and CTPS based brain components selected, e.g., based on trigger or response based components (after ICA/CTPS on cleaned data) - ```109925_CAU01A_100715_0842_2_c,rfDC,bp1-45,ar,ctpsbr-trigger-raw.fif```.
   
   (,ctpsbr-trigger)
   (,ctpsbr-response)

   If you choose a combination of different phase-locked components then simply add

   (,ctpsbr)

12. Raw file after application of noise reducer gets the postfix ',nr'. e.g. ```109925_CAU01A_100715_0842_2_c,rfDC,nr-raw.fif```

   Typically the postfix ',nr' includes noise reducer applied for the removal of 50Hz, 60Hz and harmonics and low frequencies up to   5Hz. But sometimes 'nfr' is used to indicate removal of 50/60 Hz and 'nr' is used to indicate removal of low frequency noise.
