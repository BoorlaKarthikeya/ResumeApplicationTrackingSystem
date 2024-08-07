# Resume Application Tracking system

Introducing our innovative Resume Application Tracking System (RATS)! RATS is a cutting-edge platform designed to streamline the job search process by leveraging the power of Natural Language Processing (NLP) and Machine Learning.

With RATS, users can simply upload their resume, and our advanced algorithms will analyze its content to suggest the top 5 job opportunities tailored to their skills and experience. Our system goes beyond keyword matching by deeply understanding the nuances of each resume, extracting key information, and identifying relevant job opportunities that align with the user's profile.

Powered by state-of-the-art NLP techniques, RATS not only considers job titles and descriptions but also delves into the finer details of the resume, such as skills, qualifications, and work experience. By comprehensively analyzing the resume's content, RATS ensures that the suggested job matches are highly accurate and personalized to the user's career aspirations.

## overview
The Resume Application Tracking System (ATS) is designed to streamline the recruitment process by analyzing resumes and job descriptions. This system offers two key features:

- **Department Prediction**: Predicts the most suitable department for a job seeker based on the keywords in their resume. 
- **ATS Score and Keyword Analysis:** Provides an ATS score and identifies missing keywords when comparing a resume to a job description.


## Features
1. **Department Prediction**
- **Description:** This feature uses a machine learning model to predict the department in which a candidate is most likely to get a job.
- **Model Used:** OneVsRestClassifier with KNeighborsClassifier.
How it Works:
Extracts keywords from the resume.
- Uses the classifier to predict the suitable department based on these keywords.
2. **ATS Score and Keyword Analysis**
- **Description:** This feature compares a given resume with a job description to provide an ATS score and identify missing keywords using gemini API key.
- **How it Works:**
    - Takes a resume and a job description as input.
    - Analyzes the resume to calculate an ATS score based on the job description.
    - Identifies and lists the keywords missing from the resume that are present in the job description.

 Dataset :https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
## Demo

- [**Demo Link**](https://resumeapplicationtrackingsystem-karthikeya.streamlit.app/)



Experience the future of job hunting with RATS – where finding your dream job is just a resume away!


