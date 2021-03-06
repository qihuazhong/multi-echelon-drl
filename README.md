# multi-echelon-drl
Solving Multi-Echelon Inventory Management Problems with Heuristic-Guided Deep Reinforcement Learning

Recommended Python version: 3.9

## Installation

```commandline
git clone https://github.com/qihuazhong/multi-echelon-drl.git
```

To install the required packages:
```commandline
pip install --upgrade -r requirements.txt
```

## Training DRL agents

The experiments setup and hyperparameters of the DRL algorithm should be defined in a YAML file.  Examples of experiments setup files can be found at `setup_exmaple_centralized.yaml` and `setup_exmaple_decentralized.yaml`.


## Usage 

### Commandline arguments
```
usage: train_td3.py [-h] -e E [--name NAME]

optional arguments:
  -h, --help   show this help message and exit
  -e E         Path to the experiment setup file (.yaml)
  --name NAME  Name of the experiment. Used as a prefix saving log files and models to avoid overwriting        
               previous experiment outputs
```


### Examples

To train a TD3 agent with heuristic guided exploration (HGE), simply run the train script with the path to a YAML file as the first argument:
```commandline
python train_td3.py -e setup_exmaple_centralized.yaml
```

Or train a TD3 / DQN / A2C  without HGE in the decentralized setting:

```commandline
python train_td3.py -e setup_exmaple_decentralized.yaml
python train_a2c.py -e setup_exmaple_decentralized.yaml
python train_dqn.py -e setup_exmaple_decentralized.yaml
```