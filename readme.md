# Car Segmentation using Genetic Programming

This project is about a model for car segmentation using genetic programming for the [paper](http://abricom.org.br/eventos/cbic2019/cbic2019-85/): A Gene Expression Programming Approach for Vehicle Body Segmentation and Color Recognition.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```
To install *gep* clone the repository of [geppy](https://github.com/ShuhuaGao/geppy)
```bash
git clone https://github.com/ShuhuaGao/geppy.git
```
Then you need to create a symbolic link to use ```import gep``` from the project root 
```bash
ln -s gep geppy/geppy
````

## Compile

```bash
python segmentation.py
```

## Test

For the testing process, you need a string that means the order of the operators as ```add(sub(logarithm(add(sqrt(max_(IM_9, add(divc(subc(IM_15)), min_(IM_4, IM_1)))), IM_5)), IM_2),logarithm(sub(IM_12, min_(IM_8, IM_15))))```

This string is saved in a variable for testing in *test_segmentation.py*.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Citation
Berno, B. C. S., Albini, L. A., Tasso, V. C., Benıtez, C. M. V., & Lopes, H. S. A Gene Expression Programming Approach for Vehicle Body Segmentation and Color Recognition.
