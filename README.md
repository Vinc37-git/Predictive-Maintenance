# Predictive-Maintenance

This predictive maintanance project is part of a University Project  of the course *Machine Learning in Mechanics* at the University of the Stuttgart. <br>
The models were developed by
- Fabian Moeller,
- Felix Bode,
- Vincent Hackstein.

## Dataset

The dataset, which is known as the C-MAPSS dataset, is provided by NASA.
Detailled information and some publications can be found in [NASA's Prognostics Center of Excellence](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan).

It contains sensor measurements of aircraft gas turbines, which were tested up to the point of failure. It is used as a benchmark data frame for a wide variety of networks to test their performance on predictive Maintenance Tasks. It consists of three main parts, a test set, a training set and a set containing the RUL of the engines tested in the test set. Every engine had 21 sensors connected to it while being tested, of which each sensor contributes one measurement per engine cycle.


## Models

After reviewing and processing the data, we devicided to focus on two approaches for performing the maintenance prediction:
- A classificiation model which will predict the **Maintenance Priority** of an Engine.
- A regression model which will predict the **Remaining Useful Lifetime (RUL)** of an engine.

Please take a look at the Notebooks if you wish to learn more about our project:
- `Data_Preprocessing.ipynb`
- `Classification.ipynb`
- `Regression.ipynb`
