import os
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
from bert_score import BERTScorer

class FeedbackPipeline:
    def __init__(self, api_key, models, threshold=0.97, max_steps=3):
        
        self.client = OpenAI(api_key=api_key)
        self.models = models  
        self.threshold = threshold
        self.max_steps = max_steps
        self.results = pd.DataFrame()

    # Took refernce from: https://haticeozbolat17.medium.com/text-summarization-how-to-calculate-bertscore-771a51022964
    def bert_score(self, original, updated):
        '''
        Calculates bert score that is further used for evaluating the simialarity between various generated codes.
        '''
        scorer = BERTScorer(model_type='bert-base-uncased')  
        P, R, F1 = scorer.score([updated], [original])  
        return F1.mean().item()     

    # Took reference from: https://platform.openai.com/docs/guides/text-generation/quickstart
    def initial_text(self, text, model_name):
        '''
        Creates an initial output code that is feeded back to get a feedback.
        '''
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "Convert this text in simple language that is easy to understand and keep all the information in the initial text. Just Return the simplified text and nothing else."}
                ,
                {   
                    "role": "user", "content": text
                }
            ],
            temperature=0.5)
        return response.choices[0].message.content

    def feedback_text(self, output_text, original_text, model_name):
        '''
        Generates a feedback for the given text based on how well it is simplified from the original text. This feedback function is used iteratively to get feedbacks in a loop.
        '''
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "Give feedback that improves the simplified text. just give the feedback and not the updated text"
                },
                {
                    "role": "user", "content": f"""Compare the original text: "{original_text}" with its simplified version: "{output_text}". Evaluate if the simplification maintains key information while being easier to read, and provide specific ways to improve clarity and simplicity while preserving the original meaning."""
                }
            ],
            temperature=0.5
        )
        return response.choices[0].message.content

    def improve_text(self, output_text, feedback, model_name):
        '''
        Generates improved text (simplified text) by using a given feedback and the original text.
        '''
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "Improve the text based on the given feedback and just return the updated text and nothing else."
                },
                {
                    "role": "user", "content": f"Considering this feedback: {feedback} improve the simplified text: {output_text}"
                }
            ],
            temperature=0.5
        )
        return response.choices[0].message.content

    def run(self, original_text):
    
        for model_name in self.models:    
            for iteration in range(1, self.max_steps + 1):
                current_simplified_text = self.initial_text(original_text, model_name)
                feedback = self.feedback_text(current_simplified_text, original_text, model_name)
                improved_simplified_text = self.improve_text(current_simplified_text, feedback, model_name)
                similarity = self.bert_score(original_text, improved_simplified_text)
                self.results = pd.concat([
                    self.results,
                    pd.DataFrame({
                        'Model': [model_name],
                        'Iteration': [iteration],
                        'Original text' : [original_text],
                        'Simplified_Text': [improved_simplified_text],
                        'BERTScore': [similarity],
                    })
                ], ignore_index=True)
                if similarity >= self.threshold:
                    break
                current_simplified_text = improved_simplified_text

        self.results.to_csv('Experiments')

    #Took refernece from: https://www.geeksforgeeks.org/plot-multiple-plots-in-matplotlib/
    def plot_results(self):
        '''
        Plots the Bert Score for different models.
        '''
        for model in self.models:
            data = self.results[self.results['Model'] == model]
            plt.plot(data['Iteration'], data['BERTScore'], label=model)
        plt.xlabel('Iterations')
        plt.ylabel('BERT Score')
        plt.title('Model Analysis for text simplifications')
        plt.legend(title='Model')
        plt.savefig("Results plot.jpg")
        plt.show()

    def print_results(self):
        '''
        Prints the previously generated dataframe with the results.
        '''
        print(self.results)

if __name__ == "__main__":
    prompt = input("Ask for a simplified text: ")
    api_key = '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' # Change the key please as this will be deactivated.
    models = ["gpt-4o-mini", "gpt-4o","gpt-4","gpt-4-turbo","gpt-3.5-turbo"]# we can add more models here to comapre.
    pipeline = FeedbackPipeline(api_key=api_key, models=models, threshold=0.97, max_steps=3)
    pipeline.run(prompt)
    pipeline.print_results()
    pipeline.plot_results()