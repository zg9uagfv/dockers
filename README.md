# dockers for ubuntu

1.停止所有的container

sudo docker stop $(docker ps -a -q)

2.删除所有container

sudo docker rm $(docker ps -a -q)

3.查看当前有些什么images

sudo docker images

4.删除images，通过image的id来指定删除谁

sudo docker rmi <image id>
  
5.想要删除untagged images，也就是那些id为<None>的image的话可以用
  
sudo docker rmi $(docker images | grep "^<none>" | awk "{print $3}")
  
6.要删除全部image的话

sudo docker rmi $(docker images -q)
