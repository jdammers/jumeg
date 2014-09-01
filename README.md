jumeg
=====

MEG Data Analysis at FZJ

Warning
=======

1. Please do not push any changes directly to the master branch of jdammers/jumeg. Always make a pull request, discuss and only then merge with the master branch.
2. If you have code that you want to share and are not sure how to, please put it on a gist (https://gist.github.com/) and raise an issue on Github so that it can be discussed and added.

Installation
============

1. Fork the repository by visiting https://github.com/jdammers/jumeg and clicking on the Fork button. 
2. Your own fork of the repo will be created at https://github.com/<yourname>/jumeg.
3. Clone the repository from the system shell using the command:
   git clone https://github.com/<yourname>/jumeg
4. Add it to your local python site-packages using the below commands from the same directory as above:

   ```   site_dir=`python -c'import site; print site.getusersitepackages()'` ```

   ```   present_dir=`pwd` ```
   
   ```   ln -s "${present_dir}/jumeg" "${site_dir}/jumeg"   ```
   
5. Now open the python/ipython shell and try "import jumeg" to check if it is correctly installed. 
   

Using GIT
=============

Add the main repository to your git using the below command:
   ```
   git remote add upstream https://github.com/jdammers/jumeg
   ```
Updating to the latest master version:
   ```
   git checkout master
   git pull upstream master
   ```
Making some changes and updating it to the web server: 

1. Create a new branch for your changes:
   git checkout -b <new_branch>

2. Make your changes to the files, add new files etc. 

3. Check your changes using "git status". 

4. Add these changed files to git and commit it to the web server:
   ```
   git add <file1> <file2>
   git commit -m"Some useful comments" <file1> <file2>
   git push origin <new_branch>
   ```
5. Now go to https://github.com/<yourname>/jumeg and use the "Compare and pull request" option to raise a pull request where you can discuss and finally merge the new changes to main master branch. 

6. Use ```git checkout master``` and ```git pull upstream master``` to remain on the latest changes before making your own changes. 
