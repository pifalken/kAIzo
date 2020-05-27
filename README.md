# kAIzo
LM, RNN, & VAE level generator for SMM2

## running
- LM:

- RNN:

  training example:

  ```python
  python main.py --nepochs 50 --log-interval 2000 --rnn_type RNN --game SMM2 --data data/smm2/ --cuda --h 64 --w 128
  ```

  generating new levels with (pre)training weights:

  ```python 
  python generate.py weights/mario.pt SMM2 data/smm2/ 1.0 "XX---XXXXXXXXX--------------" 120
  ```
  
- VAE:


# @TODO
- add details to README
- support LSTM
- finish VAE [!]
- optimize model code
