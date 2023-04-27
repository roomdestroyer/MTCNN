## Quick Start

Run the following command to create a new environment named "py39" and install Python 3.9.12 in it:

```shell
conda create --name py39 python=3.9.12
```

Activate the new environment:

```shell
conda activate py39
```

Now install the necessary packages and libraries in this environment, use the following commands to prepare your environment:

```shell
pip install tensorflow opencv-python tdqm matplotlib
```

> If you are running this repository under MacOS, use 'pip install tensorflow-macos' instead. And if you are having problem to connect the pip server, use the pip mirror instead, the command will be 'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-macos opencv-python tdqm matplotlib'

At this stage, download this project to your local machine:

```shell
git clone https://github.com/roomdestroyer/MTCNN.git
```

```
cd MTCNN
```

You have NO NEED to creat any directories or download any datasets manually, the whole process is integrated into python scripts. Just run the follwing command to create your directories, download datasets, generate training data, and train your models.

~~~
python main.py -all
~~~

Some other commands are well supported, like the ones listed below, you can check the `main.py` file for further useful information.

~~~
python main.py [ -create | -gen p | -gen r | -gen o | -train p | -train r | -train o | -train logs | -test imgs | -test videos | -all]
~~~



