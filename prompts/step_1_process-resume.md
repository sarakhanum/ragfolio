prompt = """
You are an expert system for converting resumes into high-quality knowledge documents optimized for Retrieval-Augmented Generation (RAG) systems and chatbot embeddings.

Your goal is NOT to rewrite or shorten the resume. Your goal is to EXPAND the information so that a chatbot can answer questions about the candidate accurately.

The output should be extremely rich in context and explanations so that even indirect questions can retrieve the correct information through semantic search.

Follow the instructions below carefully.

---

TASK

Convert the resume into a detailed knowledge document suitable for embeddings.

The output should:

1. Expand every section with detailed explanations
2. Add contextual explanations of technologies
3. Describe projects deeply
4. Explain responsibilities and contributions
5. Include likely recruiter and HR questions
6. Include technical interview questions related to the candidate's work
7. Add alternative phrasings and semantic variations of key information
8. Ensure the text is highly descriptive so embedding models capture meaning
9. Preserve the candidate's identity and contact details exactly as written in the resume:
   - full name
   - phone number(s)
   - email address(es)
   - location
   - social accounts and links (e.g., LinkedIn, GitHub, portfolio, Twitter/X, personal website)

Do NOT shorten anything. Add depth.

---

OUTPUT STRUCTURE

Create the following sections.

1. Candidate Profile  
Provide a detailed professional description of the candidate including experience level, expertise areas, and working style.

2. Professional Summary  
Write a longer narrative explaining the candidate’s career path, strengths, and types of systems they have built.

3. Work Experience (Expanded)  
For each job include:
- company  
- duration  
- responsibilities  
- technologies used  
- systems built  
- challenges solved  
- measurable outcomes  
- architectural decisions  

Explain projects in narrative form.

4. Project Deep Dive  
For each major project mentioned in the resume explain:
- problem the system solves  
- system architecture  
- technologies used  
- responsibilities of the candidate  
- interesting engineering decisions  
- impact of the project  

5. Technology Knowledge  
Explain what technologies the candidate knows and how they used them.

6. Skills Context  
Turn the skills list into descriptive paragraphs explaining the candidate's practical experience.

7. Leadership and Mentorship  
Describe leadership activities such as mentoring interns, training others, managing teams, or coordinating projects.

8. Personal Projects  
Explain each personal project with purpose, technologies used, and technical design.

9. Achievements and Awards  
Explain what the awards mean and why the candidate received them.

10. Education  
Provide context around the degree and technical background.

---

CHATBOT SUPPORT DATA

11. HR Questions and Answers  
Generate 20–30 HR questions with answers based only on the resume.

12. Technical Interview Questions  
Generate 20–30 technical questions based on the candidate’s experience.

13. Recruiter Search Queries  
Generate realistic recruiter search queries that match this candidate.

14. Semantic Variations  
Rewrite key facts in multiple ways so embeddings capture meaning.

---

RAG OPTIMIZATION RULES

- Write in natural sentences, not overly bullet-heavy  
- Use varied wording for the same facts  
- Add explanations for better semantic understanding  
- Avoid very short or vague sentences  
- Include both technical and non-technical explanations  
- Do not hallucinate new achievements  
- Only expand what exists in the resume  

---

INPUT RESUME:

Sara Khanum
Email: sarakhanum62@gmail.com
Mobile: +91 8197153926

EDUCATION
P.E.S College of Engineering Mandya, India
Information Science and Engineering; CGPA: 8.66 (till 5th Sem)

Mandvya Excellence PU College
Pre-University (PCMB)

Poorna Prajana Convent [R] Maddur, India
Secondary Education

SKILLS SUMMARY
Programming Languages: Python, C
Web Technologies: HTML, CSS
Frameworks: Django, React
Tools: Github, Uipath

ACHIEVEMENTS
Participated in Smart India Hackathon (SIH), NxtWave OpenAI Buildathon, and Google Lakecity Hackathon.
Worked in a team to solve real-world problems and build software solutions.
Gained hands-on experience in rapid application development, teamwork, and technical presentation.

PROJECTS

Quiz Application
Designed and developed a web-based quiz application using Django.
Implemented user authentication, quiz management, and score evaluation features.
Stored questions and results in a database and built a user-friendly interface.

Grocery Hut Website
Built a full-stack grocery shopping website using React (Frontend) and Django (Backend).
Implemented product listing, user login, and cart management features.
Developed REST APIs to connect frontend and backend.
Focused on responsive and user-friendly design.

UrbanNest Rental Website (Prototype/In progress)
Developed a rental property website to list and manage properties.
Implemented property listings, user authentication, and property details.
Integrated frontend and backend for smooth data handling.

CERTIFICATES
RPA Basics and Introduction to UiPath - Coursera
Automation Techniques in RPA - Coursera
ETL and Data Pipelines with Shell, Airflow and Kafka - Coursera
The Bits and Bytes of Computer - Coursera
Supervised Machine Learning: Classification – Coursera (IBM)
RPA Basics and Introduction to UiPath – Coursera (UiPath)

PERSONAL DETAILS
Languages: Kannada, Hindi, English
Hobbies: Reading novels, staying updated with media and newspapers
"""