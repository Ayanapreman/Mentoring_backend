from transformers import pipeline
classifier = pipeline("sentiment-analysis")
prediction = classifier("I neutralize using transformers. The  part is wide range of support and its neutral to use", )
print(prediction)