import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
import spacy

def main():
    st.sidebar.title("CourseCupid 💘")
    st.sidebar.subheader("A course recommender for CUHK-Shenzhen students")

    @st.cache(persist=True, allow_output_mutation=True)
    def load_data():
        data = pd.read_excel("./processed_ge_course.xlsx")
        raw_data = pd.read_excel("./raw_ge_course.xlsx")
        return data, raw_data

    # Load the data
    data, raw_data = load_data()

    # spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

    def preprocess(text):
        doc = nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop])

    # Preprocess course titles
    data['processed_title'] = data['Title'].apply(preprocess)
    data['Description'] = data['Description'].apply(preprocess)

    # Calculate the vector representation for each preprocessed title and description
    data['vector_title'] = data['processed_title'].apply(lambda x: nlp(x).vector)
    data['vector_description'] = data['Description'].apply(lambda x: nlp(x).vector)

    # Convert the vectors into matrices for cosine similarity calculation
    vector_matrix_title = pd.DataFrame(data['vector_title'].tolist())
    vector_matrix_description = pd.DataFrame(data['vector_description'].tolist())

    def recommend_course(partial_title, num_of_rec=10):
        # Preprocess and vectorize the input partial_title
        preprocessed_partial_title = preprocess(partial_title)
        input_vector = nlp(preprocessed_partial_title).vector

        # Compute cosine similarity between the input_vector and course title vectors (Model 1)
        cosine_sim_titles = cosine_similarity([input_vector], vector_matrix_title)

        # Compute cosine similarity between the input_vector and course description vectors (Model 2)
        cosine_sim_descriptions = cosine_similarity([input_vector], vector_matrix_description)

        # Calculate the average cosine similarity
        cosine_sim_avg = (cosine_sim_titles + cosine_sim_descriptions) / 2

        # Get the indices of recommended courses
        sorted_scores = sorted(enumerate(cosine_sim_avg[0]), key=lambda x: x[1], reverse=True)
        selected_course_indices = [i[0] for i in sorted_scores[:num_of_rec]]
        selected_course_scores = [i[1] for i in sorted_scores[:num_of_rec]]

        # Return the recommended courses along with their similarity scores
        result = raw_data['Code'].iloc[selected_course_indices]
        rec_df = pd.DataFrame(result)
        rec_df['Title'] = raw_data['Title'].iloc[selected_course_indices]
        rec_df['similarity_scores'] = selected_course_scores
        return rec_df

    # Create a streamlit text input for the user to input a topic
    topic = st.sidebar.text_input("Enter a topic you are interested in")

    # Default number of recommended courses to 5
    num_course = st.sidebar.slider("Number of courses to recommend", 1, 10, 5)
    rec_df = recommend_course(topic, num_of_rec=num_course)

    # Display the recommended courses
    st.markdown(" ### Recommended courses")
    if topic:
        st.success("Here are some recommended courses for you:")
        st.write(rec_df)

        # Create a list from the recommended courses
        rec_list = rec_df['Code'].tolist()

        # Create a streamlit select box for the user to select a course
        course = st.sidebar.selectbox("Select a course to see its description", rec_list)

        # Display the description of the selected course
        st.markdown("### Course description")
        if course:
            st.success(f"Here is the description for {course}:")
            st.write(raw_data[raw_data['Code'] == course]['Description'].iloc[0])
    else:
        st.warning("Please enter a topic in the sidebar")

if __name__ == "__main__":
    main()
