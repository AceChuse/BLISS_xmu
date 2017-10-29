# Brain-like Intelligent Systems(BLISS_xmu)

Hi Friends!

 # 编写这份清单的主要目：避免因为新手刚刚开始学习不知道自己需要什么而浪费大量的时间而编写的。如果你是已经入门了，我水平可能不及你，当然就不需要阅读一下的内容。

    首先，想要编写AI需要学习一种编程语言和它的集成开发环境（IDE）。近年来比较火爆的编程语言是python，同时也是常用于科学计算的语言之一。

1. 这里是python的菜鸟教程：http://www.runoob.com/python/python-basic-syntax.html

2. 安装IDE之前要先安装，python的语言包。如果你使用的是ubuntu系统这步就可以跳过了，ubuntu自带了python安装包。
    建立python环境（就是安装python语言包），有两种选择直接安装python或安装Anaconda2。不同在于Anaconda2中自带了python常用的科学计算包，能节省一些安装的麻烦。

直接安装python:http://www.runoob.com/python/python-install.html （里面有写mac和Linux中的安装，无视就可以了。）
安装Anaconda2:软件的下载地址 https://www.anaconda.com/download/ ，教程http://jingyan.baidu.com/article/7908e85c9e4725af481ad2e2.html （你可能看到下载界面不一样，这没关系，只是官网的人重做的界面而已）

至于安装python2.7还是3.6，都可以大家有人用2.7，有人用3.6，好像都用的都挺好的。

3. 推荐的IDE:eclipse
    win中安装eclipse+python的安装和配置教程：http://www.cnblogs.com/Bonker/p/3584707.html （这也只是百度的）

4. 安装必要的python包

   1)numpy。这是我们使用python做任何智能程序的基础，安装教程：http://blog.csdn.net/walkandthink/article/details/45200597 （这也是百度的）
   numpy的官方教程 https://docs.scipy.org/doc/numpy-dev/user/quickstart.html 。如果你使用的是Anaconda
   上面添加的这个库中添加的 NumPy的详细教程.docx 是一份更为简短的中文教程，如果看英文不习惯的话可以先看一下。
   到这里就可以使用numpy编写一些智能程序了。对完全没接触过的新手在机器学习方面，推荐周志华的机器学习。由于版权问题就不添加pdf了。
   对于使用python实现不同的机器学习或其他的智能算法，百度：python实现... 。 一般都能找到。
   
   2）gym。这我们用来实验AI的环境，安装在输入指令 pip install gym就可以了，如果依赖包都有的话--。
   这里gym的英文详细教程：https://github.com/openai/gym
   简短的教程在这个库中添加的 OpenAI中gym教程.pdf中。
   
   3）tensorflow。大家都知道进来深度学习带来的热潮，想必也有一定的兴趣。我们在gym中实验深度强化学习，就如字面上意思也会用的深度学习。而tensorflow是一个深度学习的框架，废话我多说，总之tensorflow挺多人使用，公认好用。这个库中添加的tensorflow_manual_cn.pdf是官方教程的翻译教程。
   tensorflow_manual_cn.pdf中有在Linux中安装的教程，在windows中安装的教程暂时没有。
   4)opencv
   
5. 深度Q学习
    论文是这个库中的 Playing Atari with Deep Reinforcement Learning.pdf 和 Human-level control through deep reinforcement learning.pdf 中，其中 Playing Atari with Deep Reinforcement Learning.pdf 是2013年发表的，Human-level control through deep reinforcement learning.pdf 是2015年发表的，是加强版。DQN是深度强化学习的前身，要学习深度强化学习应该则应该先学习DQN。
    
 # 如果以上的都学完了，虽然只学习了科研的冰山一角，但是已经可以自行研究强化学习了，开发自己的算法了。一路学到这里幸苦了。
 restore
