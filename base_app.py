"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Text Cleaning tools
from textblob import TextBlob 
import cleantext

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
st.title("Data")
st.subheader("Load Data to Classify")

with st.expander('Analyze CSV'):
	new_data = st.file_uploader('Upload a csv file')

if new_data:
	raw=pd.read_csv(new_data)
else:
	raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")


	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page

	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page

	if selection == "Prediction":
		st.info("Prediction with ML Models")

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Message","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			st.subheader("Logistic Regression Model")
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction==2:
				st.success("Statement of fact")
			if prediction==1:
				st.success("Pro Climate Change Statement")
			if prediction==0:
				st.success("Neutral")
			if prediction==-1:
				st.success("Statement Unsure if Climate Change is Real")

			st.subheader("Support Vector Machine Model")
			predictor_svm = joblib.load(open(os.path.join("resources/tfidfvect.pkl"),"rb"))
			prediction_svm = predictor_svm.predict(vect_text)
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction_svm==2:
				st.success("Statement of fact")
			if prediction_svm==1:
				st.success("Pro Climate Change Statement")
			if prediction_svm==0:
				st.success("Neutral")
			if prediction_svm==-1:
				st.success("Statement Unsure if Climate Change is Real")

			# st.success("Text Categorized as: {}".format(prediction))
	


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
