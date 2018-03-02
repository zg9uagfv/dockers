# ubuntu的docker操作

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

docker run -d -p 5000:5000 -v /home/docker_registry: /var/lib/registry registry 

3.把自己镜像的打tag并上传到本地仓库。

sudo docker tag busybox 192.168.2.114:5000/busybox  

sudo docker push 192.168.2.114:5000/busybox 


