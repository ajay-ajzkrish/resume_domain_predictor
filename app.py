import streamlit as st 
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#loading clf
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))


#cleaning function
import re
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


#mapping ducitonary
category_mapper = {6: 'Data Science',
 12: 'HR',
 0: 'Advocate',
 1: 'Arts',
 24: 'Web Designing',
 16: 'Mechanical Engineer',
 22: 'Sales',
 14: 'Health and fitness',
 5: 'Civil Engineer',
 15: 'Java Developer',
 4: 'Business Analyst',
 21: 'SAP Developer',
 2: 'Automation Testing',
 11: 'Electrical Engineering',
 18: 'Operations Manager',
 20: 'Python Developer',
 8: 'DevOps Engineer',
 17: 'Network Security Engineer',
 19: 'PMO',
 7: 'Database',
 13: 'Hadoop',
 10: 'ETL Developer',
 9: 'DotNet Developer',
 3: 'Blockchain',
 23: 'Testing'}




#web app
def main():
    st.title('Resume Designation Predictor')
    upload_file = st.file_uploader('Upload Your Resume',type = ['pdf','txt'])
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            #when utf-8 fails we do latin1
            resume_text = resume_bytes.decode('latin-1')
        cleaned_resume = cleanResume(resume_text)

        #transforming
        cleaned_resume = tfidf.transform([cleaned_resume])

        #predict
        prediction_id = clf.predict(cleaned_resume)[0]

        #converting to category string
        category_predicted = category_mapper.get(prediction_id, 'Unknown')

        st.write("You resume seems to be related to  ",category_predicted," !!")

if __name__ == "__main__":
    main()