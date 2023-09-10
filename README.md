### Environment

```
python = 3.8.12
torch = 1.10.0
cuda = 11.4
```

### Run the Codes

- Run data preprocessing code

```sh
cd ./data
python data_preparation.py
```

- Run the model code

```
python main.py --train True --eval True --e 5 --b 256 --lr 0.01 --l2_reg 0.01
```

