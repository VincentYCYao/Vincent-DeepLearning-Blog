---
layout: post
title:  "SSH remote forwarding: access internal resources from outside"
date:   2020-08-11 20:00:00 +0800
categories: linux
---



*Reference: [ssh port forwarding](https://www.ssh.com/ssh/tunneling/example)*

**Senario: A machine with static IP can serve as a jump server to access other servers within intranet.** 

**Main idea:** Assuming we have `server-A` with a static IP `<static-ip>` and `server-B` without a static IP. We can, first, access `server-B` from the same internal network via SSH. Once we are on `server-B`, as the `server-A` can be access with the `<static-ip>`, a tunnel can be create from `server-B` to `server-A`. As long as we can maintain this tunnel, the access  from `server-A` to `server-B`can be guaranteed.

**Account:** The configuration on both local and remote server should be under the `admin` account. To complete the setting, the account must be in the `sudo` group. To retain the tunnel after configuration, the account on both side can not be deleted. Upon successful configuration, other users from both side can communicate with each other.



Hereafter,  `server-B` = **remote server**, `server-A` = **local server**.

# Step1: configure ssh deamon on local server

Install ssh-host on local server  -- `server-A`

```bash
sudo apt-get install openssh-server
```

Configure SSH deamon on local server -- `server-A`

```bash
sudo vim /etc/ssh/sshd_config
```

```
# set port, default is 22, you can set your own port
Port <port-on-server-A>

# !!! must enable public key authentication on local server
PubkeyAuthentication yes

# The default is to check both .ssh/authorized_keys and .ssh/authorized_keys2
# but this is overridden so installations will only check .ssh/authorized_keys
# NOTE: .ssh/authorized_keys means $HOME/.ssh/authorized_keys
AuthorizedKeysFile      .ssh/authorized_keys
```

# Step2: longin to remote server

Install ssh-host on remote server  -- `server-B`

```bash
sudo apt-get install openssh-server
```

Configure SSH deamon on remote server -- `server-B`

```bash
sudo vim /etc/ssh/sshd_config
```

```
# set port, default is 22, you can set your own port
Port <port-on-server-B>
```

Login to remote server — either physically access or ssh login from the same internal network (below is the ssh login approach)

```bash
ssh -p <port-on-server-B> admin@<some-internal-ip>
```

# Step3: create a tunnel from remote server

Generate private-publish key pair (we use `521-bits ecdsa` algorithm here) on remote server — `server-B`

```bash
ssh-keygen -t ecdsa -b 521
```

Copy publish key to local server — `server-A`

```
ssh-copy-id -p <port-on-server-A> [-i ~/.ssh/id_ecdsa.pub] admin@<static-ip>
```

Install `autossh` on remote server

```bash
sudo apt-get install autossh
```

Create a service called `autossh`

```bash
sudo vim /lib/systemd/system/autossh.service 
```

​	(content of `autossh.service `)

```bash
[Unit]
Description=Auto SSH Tunnel
After=network-online.target

[Service]
User=admin
Type=simple

# !!!
# e.g. ExecStart=/usr/bin/autossh -M 3011 -NR 2011:localhost:11 admin@<static-ip>
ExecStart=/usr/bin/autossh -M <monitor-port> -NR <arbitrary-port-a>:localhost:<port-on-server-B> admin@<static-ip>

ExecReload=/bin/kill -HUP $MAINPID
KillMode=process
Restart=always

[Install]
WantedBy=multi-user.target
WantedBy=graphical.target
```

Enable service `NetworkManager-wait-online` on remote server 

```bash
sudo systemctl enable NetworkManager-wait-online.service 
```

Enable service `autossh` on remote server 

> Enabling simply hooks the unit into various suggested places (for example, so that the unit is automatically started on boot or when a particular kind of hardware is plugged in

```bash
sudo systemctl enable autossh
```

Start `autossh`

```bash
sudo systemctl start autossh
```

# Step4: access remote server from local server

on local server, type:

```
# this command will forward the <arbitrary-port-a> on the local server to 
# <port-on-server-B> on the remote server
# <arbitrary-port-a>: the port you chose in autossh.service on remote server
ssh <user-on-remote-server>@localhost -p <arbitrary-port-a>
```


