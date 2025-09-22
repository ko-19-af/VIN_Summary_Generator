# LLM for vehicle review generation.

A LLM model trained to produce a summary, risk score, and justification for that score based on associated with a vehicles VIN number. Various types of vehicles and their associated data
are used to train the model. This vehicle data includes: a VIN, Make, Model, Year, Price, Price_to_market, Days_on_lot, Mileage, vdp_views, and Sales_opportunities.

## LLM Model

For this task a T5 transformer model was utilised. This is because T5 is designed to handle a wide range of NLP tasks by treating them all as text-to-text problems. 
This eliminates the need for task-specific architectures because T5 converts every NLP task into a text generation task.
To formulate every task as text generation, each task is prepended with a task-specific prefix (In this case we prepended all prompys with "review:"). 
Thus this woukd encourage the model to perform reviwes on the data it was provided for training.

## Processing the data
A python script was developed to help mass produce the prompts and expected outputs "data_processing.py" and the results were stored in "model_data.csv"
When processing the data several entrys were removed to incomplete data, when engineering the prompts a single style was employed so as to have the model focus more on the that is included in the text.
When creating the resulting summary, risk rating, numerous scenarious were generated based on the various data points, while not all encompassing it provides a suitable amount of options for the model to learn from.
