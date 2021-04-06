# Introduction to Quantitative Economics
<style type="text/css">
  .reveal p {
    text-align: left;
  }
  .reveal ul {
    display: block;
  }
  .reveal ol {
    display: block;
  }
</style>
## Qualitative Economics
* **Qualitative Analysis:** Deals with the existence and uniqueness of equilibria, comparative statics, qualitative explanation of data patterns...
    
    * Difficult to provide full characterization
    
    * Results too general to have quantitative significance: i.e. $\text{corr}(x,y) >0$
    
    * Often too stylized to be applicable to the real world


## Quantitative Economics
* Uses compuatational techniques to solve and simulate a structural economic model

* Allows for a more comprehensive characterization of equilibrium

    * i.e. Can you match sign and magnitude of the correlation?

* Compuational tools allow for solving far more complex and relevant cases

    * Explore all points in parameter case rather than ones which are analytically tractable

## Quantitative Economics
* **Criticism** Results are a black box
    
    * Author's duty to provide the economic understanding
    
    * Author's duty to ensure results are not driven by bugs

* **Computers are changing economics!**

## Features of Quantitative Macro

1. Questions are about measurement, answers are numerical
    * How much of Y can be explained by X?
    
    * By how much does does policy P improve some measure of welfare  

2. Use structural theory of the economy (model) to derive quantitative implications of theory 

3. Parameters of the model are **calibrated** along some dimensions of the data and used to explain **other dimensions** 

4. Computer solves equilibrium process runs experiment to answer question

Point 3. is **controversial**, the rest are widely accepted

# Steps of Quantitative Macro

## Step 1: Pose a Question

Examples

1.  How much of $Y$ (wealth distribution) can we explain with $X$ (labor income shocks)?

2. What are the wealfare implications of policy $P$?  What is the optimal policy?

3.  How does $Q$ change if we introduce a new feature $X$ into an established model $M$?


## Step 2: Choose the Model
* Build up from an established theory: e.g. the neoclassical growth model, the NK model, the Hugget-Aiyagari model

    * Solid foundation and reference of comparison

* Keep focus on the question: Do not add unecessary features

* Do not overeach: Tradeoff between realism and the ability to compute the equilibrium

* Make sure there is enough data to discipline your model parameters!

## Step 3: Calibrating the Model

Use empirical moments used to calibrate the model should be different than the onese the model aims to account for, i.e.

* **Different Frequency:** RBC is calibrated to match "long-run" facts while evaluating business cycle facts

* **Different Moments:** In Aiyagari model, $\beta$ is set to match average wealth, but tries to account for wealth inequality

* **Different Variables:** In Aiyagari model, match *income* process but evaluating *wealth* inequality

Do not use prior studies unless mapping between model and data is the same

## Step 4: Run the Experiment

* Plan out the algorithm before start writing code

    * What kind of equations do you need to solve?

    * What functions will you need to write?

    * How will you approximate equilibrium objects?

* Use pre-packaged rutines when at all possible!

* Test your code as you work: Make sure you can solve special cases/previous work

* Flesh out main result as clearly as possible:

    * Graphs are your friends

* Perform sensitivity analysis

## Criticism of Calibration
1. Calibration is a form of *Casual Empiricism*: This need not be the case, when applicable

    * Use econometric techniques, e.g. nonlinear GMM

    * Discuss parameter identification

2. Why not use all moments instead of a subset

    * By design, not choosing parameter values to best fit data we want to explain

    * **Estimation:** What does it take to match the data?

    * **Calibaration:** How much data can we explain by restricting the model in a reasonable way?

## Goals For This Course
* Develop a computational toolset to attack interesting economic questions

* Study a sequence of topics in macroeconomics

    * How to approximate equilibrium objects

* Have tools, and sample code, to do replication project and start own research

## Programming Language

*  You can use any programming language you want: Julia, Python, Matlab, C++, C or Fortran

* My examples will be in Julia

* Why Julia:

    * Clean syntax like Python

    * Puts linear algebra first like matlab

    * JIT puts performance on same level as C,C++,Fortran