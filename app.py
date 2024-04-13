import streamlit as st
import pickle
import re
import nltk
import numpy as np
import json
import google.generativeai as genai

nltk.download("punkt")
nltk.download("stopwords")

# loading models
clf = pickle.load(open("clf.pkl", "rb"))
tfidfd = pickle.load(open("tfidf.pkl", "rb"))
genai.configure(api_key="AIzaSyCQQ1J7UIDriYefI-AA5-EFjz77P5Y2Gtw")


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


def get_related_job_profiles(resume_text):
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
        1: "Designing",
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


def get_gemini_repsonse(input):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input)
    return response.text


def ats_for_jd(text, jd):

    input_prompt = """
    Hey Act Like a skilled or very experience ATS(Application Tracking System)
    with a deep understanding of tech field,software engineering,data science ,data analyst
    and big data engineer. Your task is to evaluate the resume based on the given job description.
    You must consider the job market is very competitive and you should provide 
    best assistance for improving thr resumes. Assign the percentage Matching based 
    on Jd and
    the missing keywords with high accuracy
    resume:{text}
    description:{jd}

    I want the response in one single string having the structure
    {"JD Match":"%","MissingKeywords:[]","Profile Summary":""}
    """
    output = get_gemini_repsonse(input_prompt)
    # st.subheader(output)
    output_dict = json.loads(output)
    print(output_dict)
    jd_match = output_dict["JD Match"]

    missing_keywords = output_dict["MissingKeywords"]
    profile_summary = output_dict["Profile Summary"]

    # Displaying JD Match percentage
    st.subheader(f"JD Match: {jd_match}")

    # Displaying missing keywords
    st.subheader("Missing Keywords:")
    for keyword in missing_keywords:
        st.write(keyword)

    # Displaying profile summary
    st.subheader("Profile Summary:")
    st.write(profile_summary)


# python main
# web app
st.title("Resume Application Tracking System")
st.write(
    "Feature 1 : Our Resume Application Tracking System utilizes NLP and Machine Learning to suggest the top 5 job opportunities tailored to your resume out of 25 job profiles on which the model is trained . Simply upload your resume, and let RATS streamline your job search with personalized recommendations. Experience the future of job hunting today!"
)
st.write(
    "Feature 2: RATS enhances job matching with Google Gemini API integration. Input your job description and resume, and RATS analyzes the match, identifies missing keywords, and generates a detailed profile summary. Experience comprehensive job matching tailored to your skills and requirements."
)
uploaded_file = st.file_uploader("Upload Resume", type=["txt", "pdf"])

if uploaded_file is not None:
    try:
        resume_bytes = uploaded_file.read()
        resume_text = resume_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # If UTF-8 decoding fails, try decoding with 'latin-1'
        resume_text = resume_bytes.decode("latin-1")

    jd = st.text_area("Paste the Job Description")
    submit1 = st.button("Job profiles that suits your resume")

    submit2 = st.button("ATS score for given job description")

    if submit1:
        get_related_job_profiles(resume_text)

    if submit2:
        ats_for_jd(resume_text, jd)
