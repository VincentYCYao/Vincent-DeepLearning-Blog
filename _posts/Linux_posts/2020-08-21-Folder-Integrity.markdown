---
layout: post
title:  "Check the integrity of backup folder"
date:   2020-08-21 16:40:00 +0800
categories: linux
---



**Caveate**: Don’t rely on direct comparison of disk usage or file size. Note first that the disk usage is not the same as file size. 

# Disk usage

Depending on the partition scheme and file system type, two exact same folders may have different disk usage. Data are stored in blocks of the disk, so the minimal storage unit is block. The block size may be different for various file system and partition type. Therefore, a file with actual file size smaller than a block size will occupy the whole bolck. Thus, such small file will has different disk usage on two disk with different block size. 

Here’s an example. I create a `test.txt` file with just character “1”. 

```bash
echo '1' > test.txt
```

We can see that it take 2 bytes.

```bash
➤ ls -lh test.txt 
-rw-rw-r-- 1 admin admin 2 Jul 30 18:54 test.txt
```

Yet, the disk usage is 4K. 

```bash
➤ du -h test.txt 
4.0K    test.txt
```

That’s because the block size of the disk is 4K. We can check the disk that the file is on, then get information about the block size. 

```bash
➤ df test.txt 
Filesystem      1K-blocks       Used Available Use% Mounted on
/dev/sdb2      1920753928 1638570836 184544388  90% /
➤ sudo tune2fs -l /dev/sdb2 | grep Block
Block count:              488111616
Block size:               4096
Blocks per group:         32768
```

The block size is 4098 bytes = 4 KB.

# File size

The actual file size can be inspected by `du`.

```bash
➤ du -b test.txt 
2       test.txt
```

**Sparse file**

The program behavior would also affect the backup. One example is [sparse file](https://en.wikipedia.org/wiki/Sparse_file). Sparse file is a crude form of compression where blocks containing only null bytes are not stored. When you copy a file, the `cp` command reads and writes null bytes, so where the original has missing blocks, the copy has blocks full of null bytes.

**Trust the backup program**

Trust other developers’ works while be caution to the usage of command. As long as you read the manual carefully and use the correct options, few things could go wrong. Try this:

```bash
man rsync
```

# Calculate the checksum

If you really want to make sure that the backup folder is identical to the original one, you can try MD5 algorithm with `md5sum`, SHA-2 algorithm with `sha224sum`, `sha256sum`, `sha384sum`, `sha512sum`, or BLAKE2  algorithm with `b2sum`. 

> Do not use the MD5 algorithm for security related  purposes.

We’ll take `sha512sum` as an example. Let’s say we have backed up the folder `~/Desktop/Vt_Research` to `~/Desktop/Vt_Research_bak`. We can calculate the "checksum of checksum” — we get the checksum for each file and store the result in a list, then calculate the checksum of that list.

```bash
➤ find ~/Desktop/Vt_Research_bak -type f -exec sha512sum '{}' ';' | awk '{print $1}'| sha512sum
420724a09851e07ac3ef2811fbd896229ed95bac76cc1510f22ae4df4cee581ce2ff3f9309b9e1c25d211beafaa469204795d34e0fe0a9b1f4b9a6ea05a9bfd0  -
➤ find ~/Desktop/Vt_Research -type f -exec sha512sum '{}' ';' | awk '{print $1}'| sha512sum
420724a09851e07ac3ef2811fbd896229ed95bac76cc1510f22ae4df4cee581ce2ff3f9309b9e1c25d211beafaa469204795d34e0fe0a9b1f4b9a6ea05a9bfd0  -
```

The `awk '{print $1}’` part pick the checksums ignoring the file paths. If the file contents are identical, the output will be the same. 

Here’s another problem with this method. If we change the file name in backup folder, it will not affect the result. Assuming we have `my_PNAS_paper.pdf` in `~/Desktop/Vt_Research`. Let’s rename the file first, then do the same again.

```bash
➤ mv ~/Desktop/Vt_Research_bak/my_PNAS_paper.pdf ~/Desktop/Vt_Research_bak/my_Science_paper.pdf 
➤ find ~/Desktop/Vt_Research_bak -type f -exec sha512sum '{}' ';' | awk '{print $1}'| sha512sum
420724a09851e07ac3ef2811fbd896229ed95bac76cc1510f22ae4df4cee581ce2ff3f9309b9e1c25d211beafaa469204795d34e0fe0a9b1f4b9a6ea05a9bfd0  -
➤ find ~/Desktop/Vt_Research -type f -exec sha512sum '{}' ';' | awk '{print $1}'| sha512sum
420724a09851e07ac3ef2811fbd896229ed95bac76cc1510f22ae4df4cee581ce2ff3f9309b9e1c25d211beafaa469204795d34e0fe0a9b1f4b9a6ea05a9bfd0  -
```

It returns the same result. That’s because it does not take into account the file name in the second layer of the checksum.

If a file hyerachy is important, there is another solution. Since the folder name itself is different already, we have to get into the folder first, then do the job.

```bash
➤ cd ~/Desktop/Vt_Research
➤ find . -type f -exec sha512sum '{}' ';' | sha512sum
b0270e206fba1f50cd0ccff45a3ea270314a5fea88f12ea1dd395d1e879b3371177c8927eb39a2d7e8936d4c3f0f41b51bc6710f3091667fc5cbc8ce90a5bfeb  -
➤ cd ~/Desktop/Vt_Research_bak
➤ find . -type f -exec sha512sum '{}' ';' | sha512sum
c1bccfb898a5a4042da26fe6fd9a6a6efec6816c8b499a53037ea892829653b59d23aa738a3356dab398cae6a4d69fdeda350cfcf05fb93792e75234c681c323  -
```

Let’s rename the pdf file back to its old name, and see what happens.

```bash
➤ mv ~/Desktop/Vt_Research_bak/my_Science_paper.pdf ~/Desktop/Vt_Research_bak/my_PNAS_paper.pdf 
cd ~/Desktop/Vt_Research
➤ find . -type f -exec sha512sum '{}' ';' | sha512sum
b0270e206fba1f50cd0ccff45a3ea270314a5fea88f12ea1dd395d1e879b3371177c8927eb39a2d7e8936d4c3f0f41b51bc6710f3091667fc5cbc8ce90a5bfeb  -
➤ cd ~/Desktop/Vt_Research_bak
➤ find . -type f -exec sha512sum '{}' ';' | sha512sum
b0270e206fba1f50cd0ccff45a3ea270314a5fea88f12ea1dd395d1e879b3371177c8927eb39a2d7e8936d4c3f0f41b51bc6710f3091667fc5cbc8ce90a5bfeb  -
```

**Reminder: Do not do this on large folder**


