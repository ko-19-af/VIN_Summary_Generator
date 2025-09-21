import pandas as pd
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

# from flask import Flask

# app = Flask(__name__)

model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model_output_folder = "./models/t5_fine_tuned_reviews"
model = T5ForConditionalGeneration.from_pretrained(model_output_folder)

def generate_review(prompt):

    inputs = tokenizer("review: " + prompt, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
    # num_beams: with larger value, the model explores more possibilities (and takes longer)
    outputs = model.generate(inputs['input_ids'], max_length=128, no_repeat_ngram_size=3, num_beams=6,
                             early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def find_vin(vin):
    df = pd.read_csv('sample_data.csv')  # read in the csv with the data

    info = df.loc[df['VIN'] == vin]

    Make = str(info['Make'])
    Year = str(info['Year'])
    Cp = str(info['Current price'])
    Cpm = str(info['Current price to market %'])
    Mileage = str(info['Mileage'])
    TVDPs = str(info['Total VDPs (lifetime)'])
    TSO = str(info['Total sales opportunities (lifetime)'])
    DOL = str(info['DOL'])

    return Make, Year, Cp, Cpm, Mileage, TVDPs, TSO, DOL

# @app.route('/user/<vin>')
# def review_car(vin):
def main():
    # vin = {vin}
    vin = input("Enter VIN: ")
    Make, Year, Cp, Cpm, Mileage, TVDPs, TSO, DOL = find_vin(vin)

    result = str(generate_review("Give me a brief summary and numerical risk evaluation on a scale of (1-10), along with a short justification for that score, on a car with the following stats: "
                          "Make: " + Make +
                          "Year: " + Year +
                          "Current Price: " + Cp + " "
                          "Current Price to Market %: " + Cpm +
                          "Mileage: " + Mileage +
                          "Total VDPs: " + TVDPs +
                          "Total sales opportunities: " + TSO +
                          "DOL: " + DOL + "."))

    dict_result = {'summary': "", 'risk_rating': 0, 'reason': ""}

    result = result.split(':')

    dict_result['summary'] = result[1].replace("risk_score", '')
    dict_result['risk_rating'] = int(result[2].replace("reasoning", ''))
    dict_result['reason'] = result[3]

    dict_json = json.dumps(dict_result)
    return dict_json

if __name__ == "__main__":
    # app.run(debug=True)
    main()