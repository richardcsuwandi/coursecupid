import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spellchecker import SpellChecker
import neattext.functions as nfx
import string

def clean_df(file_path):
    # Read the excel file
    df = pd.read_excel(file_path)

    # Remove \n and \t
    df['Title'] = df['Title'].replace('\n', ' ', regex=True).replace('\t', ' ', regex=True)
    df['Description'] = df['Description'].replace('\n', ' ', regex=True).replace('\t', ' ', regex=True)

    # Remove leading and trailing spaces
    df['Title'] = df['Title'].apply(lambda x: x.strip())
    df['Description'] = df['Description'].apply(lambda x: x.strip())

    # Convert to lowercase
    df['Title'] = df['Title'].str.lower()
    df['Description'] = df['Description'].str.lower()

    # Remove punctuation, including double quotes``
    punctuation_with_quotes = string.punctuation + '“”'
    translator = str.maketrans('', '', punctuation_with_quotes)
    df['Title'] = df['Title'].apply(lambda x: x.translate(translator))
    df['Description'] = df['Description'].apply(lambda x: x.translate(translator))

    # Remove stopwords
    df['Title'] = df['Title'].apply(nfx.remove_stopwords)
    df['Description'] = df['Description'].apply(nfx.remove_stopwords)

    return df

def main():
    # Add image logo to the sidebar
    st.sidebar.image("logo.png", width=150)
    st.sidebar.title("CourseCupid 💘")
    st.sidebar.subheader("A course recommender for CUHK-Shenzhen students")

    # If english is not downloaded, download it
    # if not spacy.util.is_package("en_core_web_md"):
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

    def preprocess(text):
        # Use spacy to lemmatize the text
        doc = nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_stop])

    data_processed = True
    if data_processed == False:
        # Load and clean the raw data
        file_path = 'raw_ge_course.xlsx'
        cleaned_df = clean_df(file_path)

        # Preprocess course titles and descriptions
        processed_df = pd.DataFrame()
        processed_df['processed_title'] = cleaned_df['Title'].apply(preprocess)
        processed_df['Description'] = cleaned_df['Description'].apply(preprocess)

        # Save the processed dataframe to an excel file
        processed_df.to_excel('processed_df.xlsx', index=False)

    # Load the data
    @st.cache_resource()
    def load_data():
        data = pd.read_excel("./processed_df.xlsx")
        raw_data = pd.read_excel("./raw_ge_course.xlsx")
        return data, raw_data

    data, raw_data = load_data()

    def recommend_course(topic, num_to_rec=10):
        # Preprocess and vectorize the input topic
        spell = SpellChecker()
        corrected_topic = " ".join([spell.correction(word) for word in topic.split()])
        preprocessed_topic = preprocess(corrected_topic)
        input_vector = nlp(preprocessed_topic).vector

        # Calculate the vector representation for each preprocessed title and description
        vector_matrix_title = pd.DataFrame(data['processed_title'].apply(lambda x: nlp(x).vector).tolist())
        vector_matrix_description = pd.DataFrame(data['Description'].apply(lambda x: nlp(x).vector).tolist())

        # Compute cosine similarity between the input_vector and course title vectors (Model 1)
        cosine_sim_titles = cosine_similarity([input_vector], vector_matrix_title)

        # Compute cosine similarity between the input_vector and course description vectors (Model 2)
        cosine_sim_descriptions = cosine_similarity([input_vector], vector_matrix_description)

        # Calculate the average cosine similarity
        cosine_sim_avg = 0.5*cosine_sim_titles + 0.5*cosine_sim_descriptions

        # Get the indices of recommended courses
        sorted_scores = sorted(enumerate(cosine_sim_avg[0]), key=lambda x: x[1], reverse=True)
        selected_course_indices = [i[0] for i in sorted_scores[:num_to_rec]]
        selected_course_scores = [i[1] for i in sorted_scores[:num_to_rec]]

        # Return the recommended courses along with their similarity scores
        result = raw_data['Code'].iloc[selected_course_indices]
        rec_df = pd.DataFrame(result)
        rec_df['Title'] = raw_data['Title'].iloc[selected_course_indices]
        rec_df['Similarity Scores'] = selected_course_scores
        return rec_df

    def evaluate_recommendations(rec_df, k):
        top_k = rec_df['Code'].iloc[:k]
        top_k_vectors = pd.DataFrame(data['processed_title'].apply(lambda x: nlp(x).vector).tolist()).iloc[top_k.index]
        pairwise_similarities = cosine_similarity(top_k_vectors)
        ils = np.mean(pairwise_similarities[np.triu_indices(k, k=1)])
        return np.round(ils, 2)

    st.markdown(" ### Recommended courses")

    # Create a text input for the user to input a topic
    topic = st.sidebar.text_input("Enter a topic you are interested in")

    # Default number of recommended courses to 5
    num_to_rec = st.sidebar.slider("Number of courses to recommend", 1, 10, 5)

    # Display the recommended courses
    rec_df = recommend_course(topic, num_to_rec)

    if topic:
        st.success("Here are some recommended courses for you:")
        st.write(rec_df)

        if num_to_rec > 1:
            if st.sidebar.checkbox("Show Intra-List Similarity (ILS)"):
                ils = evaluate_recommendations(rec_df, num_to_rec)
                st.info(f"ILS for the recommended courses: {ils}")

        # Create a list from the recommended courses
        rec_list = rec_df['Code'].tolist()

        # Display the description of the selected course
        st.markdown("### Course description")
            # Create a select box for the user to select a course
        course = st.selectbox("Select a course to see its description", rec_list)
        if course:
            # st.success(f"Here is the description for {course}:")
            st.write(raw_data[raw_data['Code'] == course]['Description'].iloc[0])
    else:
        st.warning("Please enter a topic in the sidebar")

if __name__ == "__main__":
    main()
