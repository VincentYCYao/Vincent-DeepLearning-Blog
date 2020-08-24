---
layout: post
title:  "Move home directory to a new disk"
date:   2020-08-24 16:00:00 +0800
categories: Server
---

# Move home directory to a new disk

## Step 1. Identify the device to-be-mounted

**Check the mounted disk:**

```bash
➤ df -l | grep /dev/
/dev/sdb2      1920753928 1638484288 184630936  90% /
/dev/sdb1          523248       6152    517096   2% /boot/efi
```

**Find all disks installed in the machine, including the unmounted one:**

```bash
➤ sudo fdisk -l | grep /dev/
Disk /dev/sda: 7.3 TiB, 8000450330624 bytes, 15625879552 sectors
/dev/sda1   2048 13671874559 13671872512  6.4T Linux filesystem
Disk /dev/sdb: 1.8 TiB, 1999844147200 bytes, 3905945600 sectors
/dev/sdb1     2048    1050623    1048576  512M EFI System
/dev/sdb2  1050624 3905943551 3904892928  1.8T Linux filesystem
```

By comparing the outputs, we can find out the unmounted disk. 

In this example, the device `/dev/sda1` has not been mount. We will move our `/home` to `/dev/sda1` later.



## Step 2. Backup home folder

**Create a temporary home folder**

```bash
sudo mkdir /home-backup
```

**Mount the device to the home backup folder**

```bash
sudo mount /dev/sda1 /home-backup
```

Now, you can double-check that the `/dev/sda1` has been mounted to `/home-backup` :

```bash
➤ df -l | grep /dev/
/dev/sdb2      1920753928 1638484288  184630936  90% /
/dev/sdb1          523248       6152     517096   2% /boot/efi
/dev/sda1      6780991912      90140 6439088576   1% /home-backup
```

**Backup home folder**

```bash
sudo rsync -av /home/ /home-backup
```

or

```bash
sudo cp -aR /home/* /home-backup/
```

**Check the backup** 

```bash
sudo diff -r /home /home-backup
```

**Delete old home folder**

```bash
sudo rm -rf /home/*
```

**Unmount the new device**

```bash
sudo umount /home-backup
```

**Mount the new device to home folder**

```bash
sudo mount /dev/sda1 /home
```



## Step 3. Mount the new disk permanently

**Get the partition UUID**

```bash
sudo blkid | grep /dev/sda1
```

**open `/etc/fstab` & add the following line**

```
UUID=[your partition's UUID]	/home	ext4	defaults 0 2
```

