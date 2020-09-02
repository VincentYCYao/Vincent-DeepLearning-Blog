---
layout: post
title:  "PyCharm, Conda & X11-forwarding: set up remote developing environment"
date:   2020-09-02 18:45:00 +0800
categories: server
---

OS: Ubuntu 18

# Background

For deep learning research, we want to fully utilize the computing power of GPUs for our model training. One simple scenario is that we want to test our algorithm to ensure that it can run on GPUs. We can also easily develop our code that controls the data type and data communication between CPU and GPU within a GPU-environment. 

There are many approaches to test your code in a remote remote environment. 

* VSC: the ssh-terminal feature (on local machine, with GUI)
* PyCharm-Professional: the ssh-terminal feature (on local machine, with GUI)
* **PyCharm-Community + X11-forwarding**: free (launch on remote machine and display on local machine, with GUI)
* python debug command (on remote machine, without GUI)

**NOTE:**  Though Pycharm-professional is not for free, you can apply for free usage of the professional version using the university email. 

#  X11-forwarding

**X11-forwarding** enables you to launch a GUI-based application from a remote machine. To utilize the X11-forwarding feature, you have to feed the `-X` tag to ssh command when you login to the  remote machine. For example:

```
ssh -X <user-name>@<ip> -p <port-num>
```

Enable X11-forwarding: edit the `/etc/ssh/sshd.config` on remote machine

# PyCharm

 **Installation**

```bash
# Install pycharm-community
sudo snap install pycharm-community --classic
```

**Configure and launch pycharm**

```bash
# find the path, it should be sth. like </snap/bin/pycharm-community>
whereis pycharm-community
# check the $PATH to see whether /snap/bin is in $PATH
$PATH
# add path to $PATH if necessary
echo 'export PATH=/snap/bin/:$PATH' >> ~/.bashrc
source ~/.bashrc
# launch pycharm
pycharm-community
```

**NOTE:** The GUI forwarded from the server is totally the same as the one you are using from your local machine, **do not mix them up :)**. Some tricks:

* check the hostname from the terminal
* rename the configurations on the top-right corner, e.g. “xxx-Server”

**PyCharm plugins**

Explore the pycharm plugins, you’ll find your love. Here are what I am using:

* IdeaVim: Vim emulator that support vim features
* Terminal: provides integrated terminal
* Machine Learning Code Completion
* Material Theme UI: recomend the “Dracula” theme
* Git
* GitHub
* Markdown
* TeXiFy IDEA

# Conda

First thing first, whenever you create a new python project, the most important step is setting up the python interpreter. Using the python interpreter of your virtual enviornment is recommended, which can separate the develop enviroments for each of your project. 

Conda is officially recommended by PyTorch and TensorFlow. Here’s some conda command: 

(reference: [conda user guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html))

```bash
# create conda env
conda create --name <new-env-name>
# check envs
conda env list
# activate env
conda activate <env-name>
# within conda env, to deactivate
conda deactivate
# auto-activate conda env (one of many ways)
echo 'conda activate <env-name>' >> ~/.bashrc
# install package like pandas
conda install pandas
```

**Caveate:** Remenber to choose a virtual environment for your pycharm project. Go to` Setting > Project: > Python Interpreter > show all > click '+' to select your virtual env`

