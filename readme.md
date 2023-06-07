# Reinforcement learning for differential evolution

## Setup

### Making virtual environment
```
ptyhon3 -m venv venv
```

### Activating the environment
```
source venv/bin/activate
```

### Installing required packages
```
pip install -r requirements.txt
```

## Testing basic algorithms

### Differential evolution
Defaultly it uses all available cores. It can be changed through `N_JOBS` variable
```
python3 test_de.py
```

## Q Learning
```
python3 test_qlearning.py
```
