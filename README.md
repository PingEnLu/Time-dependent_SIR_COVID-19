# Time-dependent_SIR_COVID-19
This repository provides the codes for the paper "A Time-dependent SIR model for COVID-19 with Undetectable Infected Persons" by Ping-En Lu.

We have uploaded the paper to arXiv and has been published. It is available at https://arxiv.org/abs/2003.00122.
The Authors are Yi-Cheng Chen, Ping-En Lu, Cheng-Shang Chang, and Tzu-Hsuan Liu

If there is an update for the paper, the latest version of the paper will be placed on this link: http://gibbs1.ee.nthu.edu.tw/A_TIME_DEPENDENT_SIR_MODEL_FOR_COVID_19.PDF

## Abstract of the paper
In this paper, we conduct mathematical and numerical analyses to address the following important questions for COVID-19: (Q1) Is it possible to contain COVID-19? (Q2) If COVID-19 can be contained, when will be the peak of the epidemic, and when will it end? (Q3) How do the asymptomatic infections affect the spread of disease? (Q4) If COVID-19 cannot be contained, what is the ratio of the population that needs to be infected in order to achieve herd immunity? (Q5) How effective are the social distancing approaches? For (Q1) and (Q2), we propose a time-dependent susceptible-infected-recovered (SIR) model that tracks two time series: (i) the transmission rate at time ![t](https://render.githubusercontent.com/render/math?math=t) and (ii) the recovering rate at time ![t](https://render.githubusercontent.com/render/math?math=t). Such an approach is not only more adaptive than traditional static SIR models, but also more robust than direct estimation methods. Using the data provided by the National Health Commission of the People's Republic of China (NHC) [1], we show that the one-day prediction errors for the numbers of confirmed cases are almost less than 3%. Also, the turning point, defined as the day that the transmission rate is less than the recovering rate, is predicted to be Feb. 17, 2020. After that day, the basic reproduction number, known as the ![R_0](https://render.githubusercontent.com/render/math?math=R_0) value at time ![t](https://render.githubusercontent.com/render/math?math=t), is less than 1. In that case, the total number of confirmed cases is predicted to be around 80,000 cases in China under our model. For (Q3), we extend our SIR model by considering two types of infected persons: detectable infected persons and undetectable infected persons. Whether there is an outbreak in such a model is characterized by the spectral radius of a ![2 \times 2](https://render.githubusercontent.com/render/math?math=2%20%5Ctimes%202) matrix that is closely related to the basic reproduction number ![R_0](https://render.githubusercontent.com/render/math?math=R_0). We plot the phase transition diagram of an outbreak and show that there are several countries, including South Korea, Italy, and Iran, that are on the verge of COVID-19 outbreaks on Mar. 2, 2020. For (Q4), we show that herd immunity can be achieved after at least ![1-\frac{1}{R_0}](https://render.githubusercontent.com/render/math?math=1-%5Cfrac%7B1%7D%7BR_0%7D) fraction of individuals being infected and recovered from COVID-19. For (Q5), we analyze the independent cascade (IC) model for disease propagation in a random network specified by a degree distribution. By relating the propagation probabilities in the IC model to the transmission rates and recovering rates in the SIR model, we show two approaches of social distancing that can lead to a reduction of ![R_0](https://render.githubusercontent.com/render/math?math=R_0).

## Usage
### Installation
* Clone this repository.
* Enter the directory where you clone it, and run the following code in the terminal (or command prompt).
```sh
pip install -r requirements.txt
```
* Run the Python 3 code in the terminal (or command prompt).
```sh
python TimeSIR_COVID-19.py
```
### Parameters one may change/concern
* COVID-19 Dataset from different countries
* Orders of the two FIR filters in (12) an (13) in the paper
* Starting day for the data training in the ridge regression
* Other parameters in the sklearn.linear_model.Ridge of scikit-learn
* Stopping criteria of the Time-dependent SIR model
* Maximum iteration days (W in the paper) of the Time-dependent SIR model

## Future work
Note that this is a **deterministic** epidemic model based on the mean-field approximation for X(t) and R(t). Such an approximation is a result of the law of large numbers. Therefore, when X(t) and R(t) are relatively small, the mean-field approximation may not be as accurate as expected. In those cases, one might have to resort to stochastic epidemic models, such as Markov chains. We will leave it as our future work.
