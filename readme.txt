UC Professor Salary Compression Project
Econ 421 – Incentives

Project Overview
----------------
This project analyzes salary trends among professors in the University of
California (UC) system from 2010–2024. The goal is to study whether wage
compression has occurred over time and whether there are signs of increased
compression after 2022, when pay transparency became more salient in
California.

The project uses annual UC salary data across all campuses and focuses only on
professor positions.

Research Question
-----------------
Do UC professor salaries show evidence of wage compression over time, and
particularly after 2022?

Data Sources
------------
Salary data come from two public sources:

1. UC Annual Wage Database (UC Office of the President)
   https://ucannualwage.ucop.edu/wage/
   Used to collect data for 2010–2012.

2. California State Controller Government Compensation Database
   https://publicpay.ca.gov/
   Used to collect data for 2013–2024.

All data are stored as yearly CSV files and filtered to include only professor
positions.

Dataset Structure
-----------------
Each CSV contains the following columns:

Year
EmployerName
Position
RegularPay
TotalWages

Data are organized by year and stored in the professors/ directory.

Planned Methodology
-------------------
The analysis will focus on both wage levels and wage dispersion.

Key variable:
    log(TotalWages)

Planned empirical approach:

1. Clean and combine yearly salary datasets (2010–2024)
2. Classify professors by rank using the Position field
3. Estimate regressions of log wages with year and campus controls
4. Analyze wage dispersion using statistics such as variance and
   coefficient of variation
5. Compare patterns before and after 2022

The goal is to evaluate whether wage gaps between professors narrow over time
or after the increase in pay transparency salience.

Project Files
-------------
professors/
    Yearly UC professor salary CSV files

playground.ipynb
    Main analysis notebook

new_anal.ipynb
    Secondary notebook for experimentation

test_cells.py
    Script used to test individual analysis components

How to Run
----------
1. Install Python dependencies (pandas, numpy, statsmodels, scipy).
2. Open playground.ipynb in Jupyter.
3. Run cells sequentially to load data and begin analysis.

Notes
-----
The dataset currently contains nominal wage variables. Earlier attempts to
construct real wages using CPI were incorrect and will not be used in the
baseline analysis.

Authors
-------
Agnibha Bhattacharya, Jackson Wurzer, and Joshua Kenworthy
