# CerberusTS for Python
> Multivariate Timeseries Forecasting using Multi-Headed Generative Convolutional Transformers

## Background

CerberusTS is a neural network architecture and framework for performing multivariate timeseries forecasting in a generative manner. The name is inspired by the fact that this architecture uses call, response, and context(s) heads, inspired by the transformers prevalent in large language models. In this approach, the "call" head is a window of timeseries data prior to some point of prediction, the "response" head is a window of generated timeseries predictions, and the "context" heads are windows of timeseries data at different resolutions to aid in the prediction. The windows are processed using 2D convolutional layers. The size and resolution of these windows are completely customizeable by the user, with this package offering some supporting pre- and post- processing functions to do so. 

CerberusTS optionally uses Foresight, where a separate model is trained to predict multiple eventualities for the response in a single take, without using autoregressive generation, which is then embedded in the Cerberus model to aid in generative predictions, where the eventualities are used in a similar manner to contexts. The motivation of this approach is rooted in the fact that conventional transformers just predict the next probable token, which in the context of human cognition would be equivalent to starting a sentence and not knowing where it is going. While transformers have proven effectiveness, I would argue that when we begin an internal monologue we have a general idea of where we want it to end up and the keypoints we want to hit along the way. This cloud of possible eventualities collapses as the thought crystalizes. 

Additionally, CerberusTS uses a unique feature for timeseries processing I refer to as Coil Normalization, which is a reference to(and shameless plug for) my work on Self-Referencing Bayesian Fields, aka [Coils](https://github.com/ap0phasi/neuralcoil). Coil normalization translates all dynamics into a follow of probability between discrete states. Simply, every change in a feature is treated as a superposition between a maximum increase and maximum decrease. Coil normalization appears to help Cerberus in generating response predictions. 

## Usage

See the notebook here for example usage: [Climate Data Example](https://github.com/ap0phasi/cerberusPy/blob/main/tests/example_cerberus.ipynb)

## Future Development

- **Input Processing Flexibility:** Currently CerberusTS uses convolutional layers to process the various input heads. I plan on adding functionality to allow users to instead use Multi-Headed Attention. This will likely be a bit slower than convolutions but will be interesting to see if there is an improvement to performance. Similarly, I want to investigate using state space models like [Mamba](https://github.com/state-spaces/mamba) in place of attention. 

-**Coil Normalization Customization:** I plan on adding functionality that will allow users to flexibly build CerberusTS models where Coil Normalization is applied only to inputs or to outputs to better investigate its potential benefits.

-**Foresight Heteroscedastic Optimization:** Currently the approach for training the multiple eventualities of foresight relies on training multiple output channels simultaneously, but that certainly does not offer an exhaustive exploration of alternatives. I plan on testing using my [Scedastic Surrogate Swarm Optimizer](https://github.com/ap0phasi/ScedasticSurrogateSwarmPy) to train Foresight to produce eventualities that minimize heteroscedastic loss. 

_Last Updated: 2023-12-27_