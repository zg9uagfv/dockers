# ubuntu的docker操作

1.  参考:http://blog.csdn.net/bxzhu/article/details/73253032

2.  参考:https://blogs.technet.microsoft.com/machinelearning/2018/03/15/demystifying-docker-for-data-scientists-a-docker-tutorial-for-your-deep-learning-projects/

## 安装
https://docs.docker.com/install/

## 删除容器
1.停止所有的container

sudo docker stop $(sudo docker ps -a -q)

2.删除所有container

sudo docker rm $(sudo docker ps -a -q)

3.查看当前有些什么images

sudo docker images

4.删除images，通过image的id来指定删除谁

sudo docker rmi <image id>
  
5.想要删除id为<None>的image的话可以用
  
sudo docker rmi $(sudo docker images | grep "<none>" | awk '{print $3}')
  
6.要删除全部image的话

sudo docker rmi $(sudo docker images -q)

## 搭建私有仓库
1.首先下载registry镜像

sudo docker pull registry 

2.默认情况下，会将仓库存放于容器内的/var/lib/registry目录下，这样如果容器被删除，则存放于容器中的镜像也会丢失.所以我们一般情况下会指定本地一个目录（刚刚创建的/home/docker_registry）挂载到容器内的/var/lib/registry下，如下：

docker run -d -p 5000:5000 -v /xxx/xxx/docker_registry:/var/lib/registry registry 

3.把自己镜像的打tag并上传到本地仓库。

sudo docker tag busybox 192.168.2.114:5000/busybox  

sudo docker push 192.168.2.114:5000/busybox


## 提交变更

1.先退出容器然后提交变更，类似于git的commit，只提交变化的部分


sudo docker commit 3b09d0d9450(container) 192.168.2.114:5000/busybox(image)

## 备份迁移
1.通过docker save命令，将镜像保存为tar文件

sudo docker save -o ~/xxx.tar xxxx/xxxx(或者IMAGE ID)

2.把xxx.tar贝到其他容器环境中，并加载

sudo docker load -i xxx.tar

## 执行程序
sudo docker run -p 9999:9999 d09603a6f3df sh -c 'cd /opt/&&python3 ./main.py'

### 使用GPU
sudo docker run -it --gpus all 9eafd3bbbe74(image id)
