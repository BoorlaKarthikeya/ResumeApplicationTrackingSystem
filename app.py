import streamlit as st
import pickle
import re
import nltk
import numpy as np


nltk.download("punkt")
nltk.download("stopwords")

# loading models
clf = pickle.load(open("clf.pkl", "rb"))
tfidfd = pickle.load(open("tfidf.pkl", "rb"))


def clean_resume(resume_text):
    clean_text = re.sub("http\S+\s*", " ", resume_text)
    clean_text = re.sub("RT|cc", " ", clean_text)
    clean_text = re.sub("#\S+", "", clean_text)
    clean_text = re.sub("@\S+", "  ", clean_text)
    clean_text = re.sub(
        "[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", clean_text
    )
    clean_text = re.sub(r"[^\x00-\x7f]", r" ", clean_text)
    clean_text = re.sub("\s+", " ", clean_text)
    return clean_text


# web app
def main():
    st.title("Resume Application Tracking System")
    st.write(
        "Introducing RATS: Our Resume Application Tracking System utilizes NLP and Machine Learning to suggest the top 5 job opportunities tailored to your resume out of 25 job profiles on which the model is trained . Simply upload your resume, and let RATS streamline your job search with personalized recommendations. Experience the future of job hunting today!"
    )
    uploaded_file = st.file_uploader("Upload Resume", type=["txt", "pdf"])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode("latin-1")

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])

        # Make the prediction using the loaded classifier
        probabilities = clf.predict_proba(input_features)
        top_classes = np.argsort(probabilities[0])[::-1][:5]
        print(top_classes)

        # Map category IDs to category names
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        # Print the top 5 predicted categories
        cnt = 0
        for category_id in top_classes:
            cnt += 1
            category_name = category_mapping.get(category_id, "Unknown")
            st.write(
                "Predicted Category",
                cnt,
                " : ",
                category_name,
                "(",
                top_classes[cnt - 1],
                "%)",
            )
    st.markdown(
        "<br><p style='text-align: center; font-weight: bold; font-size : 20px ; background-color : red'>Developed by Karthikeya Boorla</p>",
        unsafe_allow_html=True,
    )


# python main
if __name__ == "__main__":
    main()
