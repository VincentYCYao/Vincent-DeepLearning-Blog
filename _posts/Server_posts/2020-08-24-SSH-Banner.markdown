---
layout: post
title:  "Add a banner shown before ssh login"
date:   2020-08-24 16:30:00 +0800
categories: Server
---



## Add a banner shown before ssh login

## Step 1. Configure SSH deamon

```bash
sudo nano /etc/ssh/sshd_config
```

add or edit this line: `Banner /etc/sshd_banner`

## Step 2. Edit banner

```bash
sudo touch /etc/sshd_banner
sudo nano /etc/sshd_banner
```

## Step 3. Restart SSH deamon

```bash
sudo service sshd restart
```

