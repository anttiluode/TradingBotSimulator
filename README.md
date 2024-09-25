
Trading Bot Simulator

Introduction

This is a simple, probably-not-very-effective trading bot simulator built using Streamlit. The application allows users to create trading bots that can execute basic trading strategies based on stock price data fetched from Yahoo Finance. Each bot can use different strategies such as Mean Reversion, Momentum, Moving Average Crossover, and others to simulate stock trading. However, this is just a simulator and should not be used for actual trading.

The fast fourier and wavelet bots were just based on some poop i shot with ChatGPT. Honestly, I have no idea what 01 mini put in to them. 
If you want, ask from chatgpt. Also, if you want to add other bots, ask from chatgpt. 

Also, I have not seen the bots sell so far.. So there is that.. 

Author: This project was created by Antti Luode in collaboration with ChatGPT. It is a fun and experimental project designed to simulate stock trading strategies, but it may contain numerous bugs and is not suitable for real-world trading.

Features

Add multiple bots, each with different trading strategies.
Simulate trades using stock data fetched from Yahoo Finance.
Visualize performance over time using interactive charts.
Monitor bots' portfolio allocation and gain/loss reports.

Installation

Clone the Repository:

To get started, first, clone the repository:

git clone https://github.com/anttiluode/TradingBotSimulator.git

cd TradingBotSimulator

Set Up a Virtual Environment (Recommended):

It's a good idea to create a virtual environment to manage dependencies.

Install the required packages using the provided requirements.txt file:

pip install -r requirements.txt

Run the Application:

After installing the dependencies, you can run the application using Streamlit:

streamlit run app.py

This will launch the application in your web browser. If it doesn't open automatically, navigate to http://localhost:8501.

Disclaimers
No Real Trading: This simulator is not intended for actual trading. Please do not use this simulator for real-world financial decisions.

Experimental Code: This project is still in development, and the trading strategies may not perform accurately or as expected. There may be bugs, and performance may vary.

Use at Your Own Risk: While this bot may be fun to play with, use it at your own risk. No financial responsibility is taken for the use of this bot or any decisions made based on its outputs.

Thank you for checking out the project!
