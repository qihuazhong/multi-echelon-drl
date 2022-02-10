# multi-echelon-drl
Solving Multi-Echelon Inventory Management Problems with Heuristic-Guided Deep Reinforcement Learning

Recommended Python version: 3.9
## Usage Examples

Samples of experiments setup files can be found at `setup_exmaple_centralized.yaml` and `setup_exmaple_decentralized.yaml`

To train a TD3 agent with heuristic guided exploration, simply run the train script 
```commandline
python train_td3.py setup_exmaple_centralized.yaml
```

Or train a TD3 / DQN / A2C agent in the decentralized setting:

```commandline
python train_td3.py setup_exmaple_decentralized.yaml
python train_a2c.py setup_exmaple_decentralized.yaml
python train_dqn.py setup_exmaple_decentralized.yaml
```