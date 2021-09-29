### 请遵照以下步骤操作镜像
在使用宿主机进入image目录，并运行以下命令：
```bash
nvidia-docker load < bdci509_tmg_image.tar
cd ..
nvidia-docker run -it -v `pwd`/data:/data new_image bash
```
进入镜像后，需要运行以下命令在/data中生成result.csv：
```bash
sh run.sh
```

### 镜像介绍：
镜像名称：pytorch/pytorch: 1.9.0-cuda10.2-cudnn7-runtime
|软件名称|版本|
|:-:|:-:|
|sentence_transformers||
|pandas||