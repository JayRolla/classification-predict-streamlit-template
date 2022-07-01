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

# Data dependencies
import pandas as pd

# Vectorizer
#news_vectorizer = open("resources/tfidfvect.pkl","rb")
news_vectorizer = open("resources/mytfidf_vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train_stream2.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Dolph Analytics App")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Our team has the task of predicting whether a tweet supports, refutes, or is undecided about man-made climate change. Also, it can detect if a tweet was just a news item.")
		st.markdown("Members of the Dolph Analytics team include Justice Oyemike (Business Analyst), Molapo Kgarose (Data Scientist), Olamide Oladipo (Machine Learning Engineer), David Odimegwu (Product Manager), and Rachael Njuguna (Software Engineer). ")
		
		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.subheader("Some EDA of the Data")
		st.markdown("The data is a multi-class target variables with 4 labels for sentiments. The labels are 1 for Pro or supports climate change, 2 for New tweet, 0 for undecided about climate change and -1 for denies climate change. ")
		st.markdown("There was noticeable class imbalance with the 1 (Pro) label having higher data points. The image below illustrates this.")	
		st.image("images/sentiment_labels.png", caption="Count of labels in the dataset")
		st.markdown("We also found the hashtags very informative. There were distinctive hashtags for each label. We noticed that some hashtags were generalized across all the labels. Examples were climate, climatechange, and globalwarming. So we removed these because they would skew the modeling process.")
		st.markdown("Here are some visualizations of the hashtags we found insightful in each label.")
		st.image("images/pro_hashtags.png", caption="Top hashtags for the Pro (1) Label")
		st.image("images/anti_hashtags.png", caption="Top hashtags used by people against climate change. Anti (-1) group.")
		st.image("images/neutral_hashtags.png", caption="Top hashtags used by people undecided about climate change. Neutral (0) group.")
		st.image("images/news_hashtags.png", caption="Top hashtags used when tweeting about a news item. News (2) Label.")
	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models. We are using three models: LinearSVC, Bagging Classifier and Logistic Regression")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		# first button, linear svc
		if st.button("Classify With LinearSVC"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			predictor = joblib.load(open(os.path.join("resources/linersvc.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == 1: 
				result = "1: Tweet believes in man-made Climate Change."
			elif prediction	== 2:
				result = "2: Tweet is a news item."
			elif prediction == 0:
				result = "0: Tweet neither supports or refutes believe in climate change."
			else:
				result = "-1: Tweet does not believe in man-made climate change."
			st.success("Text Categorized as {}".format(result))


		# second button, Bagging Classifier
		if st.button("Classify With Bagging Classifier"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			predictor = joblib.load(open(os.path.join("resources/bagging_classifier.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == 1: 
				result = "1: Tweet believes in man-made Climate Change."
			elif prediction	== 2:
				result = "2: Tweet is a news item."
			elif prediction == 0:
				result = "0: Tweet neither supports or refutes believe in climate change."
			else:
				result = "-1: Tweet does not believe in man-made climate change."
			st.success("Text Categorized as {}".format(result))
		
		# the third button, logistic regression button	
		if st.button("Classify With Resampled Logistic Regression"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			#predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			predictor = joblib.load(open(os.path.join("resources/logistic_regression_resampled.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if prediction == 1: 
				result = "1: Tweet believes in man-made Climate Change."
			elif prediction	== 2:
				result = "2: Tweet is a news item."
			elif prediction == 0:
				result = "0: Tweet neither supports or refutes believe in climate change."
			else:
				result = "-1: Tweet does not believe in man-made climate change."		
			st.success("Text Categorized as {}".format(result))	

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
